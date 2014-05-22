#include <iostream>
#include <cpp/dev_random.cpp>
#include <tclap/CmdLine.h>
#include <itpp/itbase.h>
#include <itpp/stat/histogram.h>
#include "cpp/RMT.cpp"	
#include <cpp/itpp_ext_math.cpp>
#include <cpp/spinchain.cpp>
#include <itpp/stat/misc_stat.h>
#include <fstream>
#include <cuda.h>
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "ev_routines.cu"
#include "cfp_routines.cu"

//using namespace std; 
//using namespace itpp;
//using namespace itppextmath;
//using namespace cfpmath;
//using namespace spinchain;


TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"normalito", "string",cmd);
TCLAP::ValueArg<unsigned int> seed("s","seed", "Random seed [0 for urandom]",false, 243243,"unsigned int",cmd);
TCLAP::ValueArg<int> qubits("q","qubits", "number of qubits",false, 4,"int",cmd);
TCLAP::ValueArg<double> J("J","ising_coupling", "Ising interaction in the z-direction",false, 1.0,"double",cmd);
TCLAP::ValueArg<double> bx("","bx", "Magnetic field in x direction",false, 1.4,"double",cmd);
TCLAP::ValueArg<double> by("","by", "Magnetic field in y direction",false, 0.,"double",cmd);
TCLAP::ValueArg<double> bz("","bz", "Magnetic field in z direction",false, 1.4,"double",cmd);
TCLAP::ValueArg<double> theta("","theta", "polar angle",false, 1.0,"double",cmd);
TCLAP::ValueArg<double> phi("","phi", "azimultal angle",false, 1.0,"double",cmd);
TCLAP::ValueArg<double> deltabx("","deltabx", "perturbation",false, 0.1,"double",cmd);
TCLAP::ValueArg<int> steps("","steps","steps",false, 100,"int",cmd);
TCLAP::ValueArg<double> Jpert("","Jpert","Perturbation on Ising",false, 0.0,"double",cmd);
TCLAP::ValueArg<int> dev("","dev", "Gpu to be used, 0 for k20, 1 for c20",false, 1,"int",cmd);


int main(int argc, char* argv[])
{

cmd.parse( argc, argv );
cout.precision(17);
cudaSetDevice(dev.getValue());

// {{{ Set seed for random
unsigned int semilla=seed.getValue();
if (semilla == 0){
  Random semilla_uran; semilla=semilla_uran.strong();
} 
itpp::RNG_reset(semilla);
// }}}

itpp::vec b(3), bpert(3); 
b(0)=bx.getValue(); 
b(1)=by.getValue();
b(2)=bz.getValue();
bpert=b;
bpert(0)=b(0)+deltabx.getValue();
string option=optionArg.getValue();

itpp::cvec state, staterev, qustate;

//ofstream fidelity;
//fidelity.open("fidelity.dat");

//qustate=RandomState(64);

//int dim=pow_2(qubits.getValue());

qustate=itppextmath::BlochToQubit(theta.getValue(),phi.getValue());

//qustate=RandomState(2);

//for(int i=0; i<qubits.getValue()+1;i++){

//list(i)=qustate;

//}

if(option=="normalito")
	state=itppextmath::TensorPow(qustate,qubits.getValue());
	
if(option=="randU")
	state=RMT::RandomCUE(pow(2, qubits.getValue()))*itppextmath::TensorPow(qustate,qubits.getValue());
	
if(option=="klimov")
	state=itppextmath::TensorProduct(itppextmath::TensorProduct(itppextmath::TensorPow(qustate,3),itppextmath::sigma(1)*qustate),itppextmath::TensorPow(qustate,qubits.getValue()-4));
	
if(option=="klimovy")
	state=itppextmath::TensorProduct(itppextmath::TensorProduct(itppextmath::TensorPow(qustate,3),itppextmath::sigma(2)*qustate),itppextmath::TensorPow(qustate,qubits.getValue()-4));
	
if(option=="klimov2")
		state=itppextmath::TensorProduct(itppextmath::TensorProduct(itppextmath::TensorPow(qustate,2),itppextmath::TensorPow(itppextmath::sigma(1)*qustate,2)),itppextmath::TensorPow(qustate,qubits.getValue()-4));


//cout<< qustate ;

staterev=state;

double Jrev=J.getValue()+Jpert.getValue();

itpp::vec list(steps.getValue());

for(int i=0;i<steps.getValue();i++){

list(i)=pow( abs( dot( conj(staterev),state)),2);

//cout<< pow( abs( dot( conj(staterev),state)),2) <<endl;

cout << list(i) <<endl;
// cout<< i<< " " << list(i) <<endl;

list(i)=sqrt(list(i));

itppcuda::apply_floquet(state, J.getValue(), b);

itppcuda::apply_floquet(staterev, Jrev, bpert); 

//cout<<abs(dot(conj(staterev),state))<<endl;

//fidelity<<pow(abs(dot(conj(staterev),state)),2)<<endl;

}
 
//fidelity.close();

//cout << staterev;

std::cout<< itppextmath::sum_positive_derivatives(list)<< endl;

//std::cout<<state<<endl;


}
