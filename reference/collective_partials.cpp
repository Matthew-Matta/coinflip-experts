// Built against the isolated committed port headers/library, never working-tree SIMD code.
#include "io.hpp"
#include "numerics.hpp"
#include <filesystem>
#include <iostream>
using namespace coinflip;
int main(int argc,char**argv){
 if(argc!=4) return 2;
 std::string ex=argv[1],base=argv[2],out=argv[3];
 auto c=json_read_file(ex+"/config.json");int H=c.at("hidden_size").num,I=c.at("intermediate_size").num,K=c.at("top_k").num;
 auto input=npy_read(ex+"/inputs/o_proj_in.npy"),acts=npy_read(base+"/expert_act.npy"),ids=npy_read(base+"/topk_ids.npy");
 int T=input.shape[0];
 auto ow=npy_read_f32(ex+"/weights/o_proj.npy",{(size_t)H,(size_t)H});
 auto dw=npy_read_f32(ex+"/weights/down_proj.npy",{(size_t)c.at("num_experts").num,(size_t)H,(size_t)I});
 AccSpec acc;
 for(int n:{2,4})for(auto wire:{DType::FP32,DType::BF16})for(auto target:{std::string("oproj"),std::string("down")}){
  int slots=target=="down"?K:1,cols=target=="down"?I:H,shard=(cols+n-1)/n;
  size_t size=(size_t)T*slots*H;std::vector<double>parts(n*size),ring(size),tree(size);
  for(int t=0;t<T;++t)for(int s=0;s<slots;++s){
   auto W=target=="down"?dw.data()+(size_t)ids.data[t*K+s]*H*I:ow.data();
   auto x=target=="down"?acts.data.data()+((size_t)t*K+s)*I:input.data.data()+(size_t)t*H;
   size_t offset=((size_t)t*slots+s)*H;
   for(int r=0;r<H;++r)for(int rank=0;rank<n;++rank){int lo=std::min(rank*shard,cols),hi=std::min(lo+shard,cols);
    parts[rank*size+offset+r]=round_to(wire,dot(W+(size_t)r*cols,x,lo,hi,acc));}
   matvec(W,H,cols,x,ring.data()+offset,acc,DType::FP32,{n,Topology::Ring,wire});
   matvec(W,H,cols,x,tree.data()+offset,acc,DType::FP32,{n,Topology::Tree,wire});
  }
  std::string prefix=out+"/"+target+"-n"+std::to_string(n)+"-"+to_string(wire);
  npy_write_f32(prefix+"-partials.npy",{(size_t)n,(size_t)T,(size_t)slots,(size_t)H},parts.data());
  npy_write_f32(prefix+"-ring.npy",{(size_t)T,(size_t)slots,(size_t)H},ring.data());
  npy_write_f32(prefix+"-tree.npy",{(size_t)T,(size_t)slots,(size_t)H},tree.data());
  std::cout<<"PASS "<<prefix<<std::endl;
 }
}
