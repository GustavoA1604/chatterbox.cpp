#include "supertonic_internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor { std::vector<float> data; int64_t ne[4] = {1,1,1,1}; };

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }
inline float mish(float x) { return x * std::tanh(std::log1pf(std::exp(x))); }

void dense(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
           int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[ic];
        y[oc] = sum;
    }
}

void dense_matmul_vec(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
                      int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += x[ic] * w.data[(size_t)ic * OC + oc];
        y[oc] = sum;
    }
}

void dense_matmul_time(const std::vector<float> & x, int L, int IC,
                       const f32_tensor & w, const f32_tensor & b,
                       int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += x[(size_t)t*IC + ic] * w.data[(size_t)ic*OC + oc];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

void conv1x1(const std::vector<float> & x, int L, int IC,
             const f32_tensor & w, const f32_tensor * b, int OC,
             std::vector<float> & y) {
    y.assign((size_t)L*OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t)oc*IC + ic] * x[(size_t)t*IC + ic];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

void depthwise_same(const std::vector<float> & x, int L, int C, const f32_tensor & w,
                    const f32_tensor & b, int K, int dilation, std::vector<float> & y) {
    y.assign((size_t)L*C, 0.0f);
    int pad_left = ((K - 1) * dilation) / 2;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            for (int k = 0; k < K; ++k) {
                int st = t + k*dilation - pad_left;
                st = std::max(0, std::min(L - 1, st));
                sum += w.data[(size_t)c*K + k] * x[(size_t)st*C + c];
            }
            y[(size_t)t*C + c] = sum;
        }
    }
}

void layer_norm(std::vector<float> & x, int L, int C, const f32_tensor & g, const f32_tensor & b) {
    for (int t = 0; t < L; ++t) {
        float mean = 0;
        for (int c = 0; c < C; ++c) mean += x[(size_t)t*C+c];
        mean /= (float)C;
        float var = 0;
        for (int c = 0; c < C; ++c) { float d=x[(size_t)t*C+c]-mean; var += d*d; }
        float inv = 1.0f/std::sqrt(var/(float)C + 1e-6f);
        for (int c = 0; c < C; ++c) x[(size_t)t*C+c] = (x[(size_t)t*C+c]-mean)*inv*g.data[c]+b.data[c];
    }
}

void convnext(const supertonic_model & m, const std::string & p, std::vector<float> & x, int L, int C, int dilation) {
    auto dw_w=read_f32(m,p+".dwconv.weight"), dw_b=read_f32(m,p+".dwconv.bias");
    auto ln_g=read_f32(m,p+".norm.norm.weight"), ln_b=read_f32(m,p+".norm.norm.bias");
    auto pw1_w=read_f32(m,p+".pwconv1.weight"), pw1_b=read_f32(m,p+".pwconv1.bias");
    auto pw2_w=read_f32(m,p+".pwconv2.weight"), pw2_b=read_f32(m,p+".pwconv2.bias");
    auto gamma=read_f32(m,p+".gamma");
    std::vector<float> residual=x,y,z;
    depthwise_same(x,L,C,dw_w,dw_b,(int)dw_w.ne[0],dilation,y);
    layer_norm(y,L,C,ln_g,ln_b);
    conv1x1(y,L,C,pw1_w,&pw1_b,(int)pw1_w.ne[2],z);
    for(float &v:z) v=gelu(v);
    conv1x1(z,L,(int)pw1_w.ne[2],pw2_w,&pw2_b,C,y);
    for(size_t i=0;i<x.size();++i){ int c=(int)(i%C); x[i]=residual[i]+gamma.data[c]*y[i]; }
}

std::vector<float> time_embedding(const supertonic_model & m, int current, int total) {
    const int D=64, half=32;
    float t = (float)current / (float)std::max(1,total);
    std::vector<float> emb(D);
    float denom = std::log(10000.0f)/(float)(half-1);
    for(int i=0;i<half;++i){ float f=std::exp((float)i * -denom); float a=t*1000.0f*f; emb[i]=std::sin(a); emb[half+i]=std::cos(a); }
    std::vector<float> h,o;
    dense(emb, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.bias"),64,256,h);
    for(float &v:h) v=mish(v);
    dense(h, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.bias"),256,64,o);
    return o;
}

void apply_rope(std::vector<float> & x, int L, int H, int D) {
    int half = D/2;
    for(int h=0;h<H;++h) for(int t=0;t<L;++t) for(int d=0;d<half;++d) {
        float theta = std::pow(10000.0f, -(float)d/(float)half);
        float angle = ((float)t/(float)L)*theta;
        float cs=std::cos(angle), sn=std::sin(angle);
        size_t i1=((size_t)t*H+h)*D+d, i2=((size_t)t*H+h)*D+half+d;
        float a=x[i1], b=x[i2];
        x[i1]=a*cs-b*sn; x[i2]=b*cs+a*sn;
    }
}

void rope_attn(const supertonic_model & m, int group, std::vector<float> & x, int L,
               const float * text_emb, int LT, std::vector<float> & out) {
    static const int qids[4]={3101,3146,3191,3236}, kids[4]={3102,3147,3192,3237}, vids[4]={3103,3148,3193,3238}, oids[4]={3110,3155,3200,3245};
    int C=512, A=256, H=4, D=64;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+3)+".attn.";
    std::vector<float> q,k,v;
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    std::vector<float> text_lc((size_t)LT*256);
    for(int t=0;t<LT;++t) for(int c=0;c<256;++c) text_lc[(size_t)t*256+c]=text_emb[(size_t)c*LT+t];
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    apply_rope(q,L,H,D); apply_rope(k,LT,H,D);
    std::vector<float> attn_out((size_t)L*A,0), scores(LT), probs(LT);
    float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LT;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[((size_t)qi*H+h)*D+d]*k[((size_t)kj*H+h)*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LT;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LT;++kj) sum+=(probs[kj]/den)*v[((size_t)kj*H+h)*D+d]; attn_out[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(attn_out,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

void style_attn(const supertonic_model & m, int group, std::vector<float> & x, int L, const float * style_ttl, std::vector<float> & out) {
    static const int qids[4]={3116,3161,3206,3251}, kids[4]={3117,3162,3207,3252}, vids[4]={3118,3163,3208,3253}, oids[4]={3119,3164,3209,3254};
    int C=512,A=256,H=2,D=128,LC=50;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+5)+".attention.";
    std::vector<float> q,k,v,ctx((size_t)LC*256),kctx((size_t)LC*256);
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) ctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
    auto kconst=read_f32(m,"vector_estimator:/Expand_output_0");
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) kctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    dense_matmul_time(kctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    for(float &vv:k) vv=std::tanh(vv);
    dense_matmul_time(ctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    std::vector<float> merged((size_t)L*A,0), scores(LC), probs(LC); float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LC;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[(size_t)qi*A+h*D+d]*k[(size_t)kj*A+h*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LC;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LC;++kj) sum+=(probs[kj]/den)*v[(size_t)kj*A+h*D+d]; merged[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(merged,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

} // namespace

bool supertonic_vector_step_cpu(const supertonic_model & model, const float * noisy_latent,
                                int latent_len, const float * text_emb, int text_len,
                                const float * style_ttl, const float * latent_mask,
                                int current_step, int total_steps,
                                std::vector<float> & next_latent_out, std::string * error) {
    try {
        int L=latent_len,Cin=144,C=512;
        std::vector<float> in((size_t)L*Cin);
        for(int t=0;t<L;++t) for(int c=0;c<Cin;++c) in[(size_t)t*Cin+c]=noisy_latent[(size_t)c*L+t];
        std::vector<float> x;
        conv1x1(in,L,Cin,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),nullptr,C,x);
        for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]*=latent_mask[t];
        std::vector<float> te=time_embedding(model,current_step,total_steps);
        static const int time_ids[4]={3095,3140,3185,3230};
        for(int group=0;group<4;++group){
            int ob=group*6;
            int dils[4]={1,2,4,8};
            for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob)+".convnext."+std::to_string(j),x,L,C,dils[j]);
            std::vector<float> tb;
            dense_matmul_vec(te,read_f32(model,"vector_estimator:onnx::MatMul_"+std::to_string(time_ids[group])),
                             read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+1)+".linear.linear.bias"),64,C,tb);
            for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]+=tb[c];
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+2)+".convnext.0",x,L,C,1);
            std::vector<float> a; rope_attn(model,group,x,L,text_emb,text_len,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.bias"));
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+4)+".convnext.0",x,L,C,1);
            style_attn(model,group,x,L,style_ttl,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.bias"));
        }
        for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.last_convnext.convnext."+std::to_string(j),x,L,C,1);
        std::vector<float> v;
        conv1x1(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),nullptr,Cin,v);
        next_latent_out.assign((size_t)Cin*L,0.0f);
        for(int c=0;c<Cin;++c) for(int t=0;t<L;++t) {
            float vel=v[(size_t)t*Cin+c]*latent_mask[t];
            next_latent_out[(size_t)c*L+t]=noisy_latent[(size_t)c*L+t]+vel/(float)total_steps;
        }
        if(error) error->clear(); return true;
    } catch(const std::exception &e){ if(error)*error=e.what(); return false; }
}

} // namespace tts_cpp::supertonic::detail
