// Native state checks: checkpoint replay, reset, partial shortconv rollback, and KV page growth.
#include "q35_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void ok(const std::string & e) { if (!e.empty()) throw std::runtime_error(e); }
static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !memcmp(a.data(), b.data(), a.size()*sizeof(float));
}
static double kl(const float * a, const float * b, int v) {
    const float am=*std::max_element(a,a+v), bm=*std::max_element(b,b+v);
    double az=0,bz=0,out=0; for(int i=0;i<v;++i) {az+=exp(a[i]-am);bz+=exp(b[i]-bm);}
    for(int i=0;i<v;++i) {double p=exp(a[i]-am)/az;out+=p*((a[i]-am-log(az))-(b[i]-bm-log(bz)));}
    return out;
}
int main(int argc,char **argv) {
    try {
        if(argc!=3) throw std::runtime_error("usage: lfm25-state MODEL TOKENS");
        FILE *f=fopen(argv[2],"rb"); if(!f) throw std::runtime_error("tokens missing");
        uint32_t np=0,ng=0; if(fread(&np,4,1,f)!=1||fread(&ng,4,1,f)!=1||ng<12) throw std::runtime_error("short fixture");
        std::vector<int32_t> ids(np+ng); if(fread(ids.data(),4,ids.size(),f)!=ids.size()) throw std::runtime_error("short ids"); fclose(f);
        q35::Options o;o.gguf=argv[1];o.n_ctx=4096;o.n_ubatch=512;o.n_rs_seq=3;o.pool=true;o.kv_initial=256;o.kv_step=256;
        q35::Engine e;ok(e.load(o));if(!e.hp().lfm2) throw std::runtime_error("LFM2 fixture required");
        std::vector<int32_t> prompt(ids.begin(),ids.begin()+np); std::vector<float> a,b,h;
        ok(e.decode(prompt,&a));const auto first=a;e.checkpoint();
        auto replay=[&](std::vector<float>& lg) {for(int i=0;i<8;++i) ok(e.decode({ids[np+i]},&lg));};
        replay(a);ok(e.rollback_to(np));replay(b);if(!same(a,b)) throw std::runtime_error("checkpoint replay differs");
        e.reset();ok(e.decode(prompt,&b));if(!same(first,b)) throw std::runtime_error("reset differs");
        std::vector<int32_t> four(ids.begin()+np,ids.begin()+np+4),tail(four.begin()+2,four.end());
        ok(e.decode_all(four,&a,&h));ok(e.rollback(np+2));ok(e.decode_all(tail,&b,&h));
        const int v=e.hp().n_vocab;double maxkl=0;for(int i=0;i<2;++i) maxkl=std::max(maxkl,kl(a.data()+(i+2)*v,b.data()+i*v,v));
        if(!std::isfinite(maxkl)||maxkl>1e-3) throw std::runtime_error("shortconv partial rollback diverges");
        e.reset();std::vector<int32_t> longp;while(longp.size()<640) longp.insert(longp.end(),prompt.begin(),prompt.end());longp.resize(640);
        ok(e.decode(longp,&a));if(e.kv_cap()<640||e.n_past()!=640) throw std::runtime_error("KV pager did not grow");
        e.checkpoint();replay(a);ok(e.rollback_to(640));replay(b);if(!same(a,b)) throw std::runtime_error("paged checkpoint differs");
        e.reset();if(e.kv_cap()!=256||e.n_past()!=0) throw std::runtime_error("KV pager reset failed");
        puts("{\"checkpoint_replay\":true,\"reset\":true,\"kv_growth\":true,\"paged_checkpoint\":true,\"kv_reset\":true}");
        printf("{\"partial_rollback_max_kl\":%.12g}\n",maxkl);
    } catch(const std::exception&e) {fprintf(stderr,"state check: %s\n",e.what());return 1;}
}
