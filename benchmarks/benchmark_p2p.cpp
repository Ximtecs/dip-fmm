// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <random>
#include <iostream>
#include "cdfmm/operators.hpp"
int main(){ using namespace cdfmm; std::mt19937 g(1); std::uniform_real_distribution<double>d(-1,1); std::vector<Vec3> xs(10000),ms(10000); for(int i=0;i<10000;++i){xs[i]={d(g),d(g),d(g)};ms[i]={d(g),d(g),d(g)};} auto t0=std::chrono::high_resolution_clock::now(); auto r=p2p_dipole_sum({0.3,-0.2,0.1},xs,ms); auto t1=std::chrono::high_resolution_clock::now(); std::cout<<r.H.x<<" "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<" ms\n"; }
