#include <iostream>
#include <fstream>
#include <itpp/itbase.h>
#include <itpp_ext_math.cpp>
#include <math.h>
#include <tclap/CmdLine.h>
#include <device_functions.h>
#include <cuda.h>
#include "tools.cpp"
#include <spinchain.cpp>
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "ev_routines.cu"
#include "cfp_routines.cu"
#include <tclap/CmdLine.h>




  TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
  TCLAP::ValueArg<int> qubits("q","qubits", "Number of qubits",false, 3,"int",cmd);
  TCLAP::ValueArg<double> ising("","ising_z", "Ising interaction in the z-direction",false, 1,"double",cmd);
  TCLAP::ValueArg<double> k("","k", "qusimomentum number",false,0,"double",cmd);
  TCLAP::ValueArg<int> dev("","dev", "Gpu to be used, 0 for k20, 1 for c20",false, 0,"int",cmd);
  TCLAP::ValueArg<int> symx("","symx", "If simetry on sigma_x is to be used ",false, 0,"int",cmd);	
  
    int main(int argc,char* argv[]) {
		
      cout.precision(17);
      cudaSetDevice(dev.getValue());
      itpp::RNG_randomize();
      cmd.parse(argc,argv);
      
      itpp::cmat vec=evcuda::invariant_vectors(qubits.getValue(),qubits.getValue(), k.getValue(), 0, symx.getValue());
      
      //int dim=itpp::rank(vec);
      cout<<vec<<endl;
      //cout<< dim<<" perame "<<endl;
      //cout<< vec<<" perame "<<endl;
      //cout<< vec(1,1)<<" perame otra vez"<<endl;
     // for (int i=0; i<vec.rows(); i++){
      //for (int j=0; j<vec.cols(); j++)
	//			cout<< itppextmath::Chop(real(vec(i,j))) <<" "<< itppextmath::Chop(imag(vec(i,j)))<<endl;
//	}
   //pow(qubits.getValue(),2)   
  }
