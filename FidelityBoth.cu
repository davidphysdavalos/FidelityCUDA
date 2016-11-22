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

/*
using namespace std; 
using namespace itpp;
using namespace itppextmath;
using namespace cfpmath;
using namespace spinchain;
using namespace RMT;
/**/

TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
TCLAP::ValueArg<string> optionArg("o","option", "Option" ,false,"normalito", "string",cmd);
TCLAP::ValueArg<string> optionArg2("","option2", "Option2" ,false,"fidelity", "string",cmd);
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
TCLAP::ValueArg<double> Jinhompert("","Jinhompert","Inhomogeneous perturbation on Ising on 0-1 interaction",false, 0.0,"double",cmd);
TCLAP::ValueArg<double> deltabxinhom("","deltabxinhom", "perturbation al campo solo en el qubit 0",false, 0.0,"double",cmd);


int main(int argc, char* argv[])
{

cmd.parse( argc, argv );
cout.precision(12);
cudaSetDevice(dev.getValue());

// {{{ Set seed for random
unsigned int semilla=seed.getValue();
if (semilla == 0){
  Random semilla_uran; semilla=semilla_uran.strong();
} 
itpp::RNG_reset(semilla);
// }}}

itpp::vec b(3), bpert(3), bpertrev(3), binhom(3), binhomrev(3);
b(0)=bx.getValue(); 
b(1)=by.getValue();
b(2)=bz.getValue();
bpert=b;
bpertrev=b;
bpert(0)=b(0)+deltabx.getValue();
bpertrev(0)=b(0)-deltabx.getValue();
binhom=bpert;
binhomrev=bpertrev;
binhom(0)=bpert(0)+deltabxinhom.getValue();
binhomrev(0)=bpertrev(0)-deltabxinhom.getValue();
string option=optionArg.getValue();
string option2=optionArg2.getValue();

itpp::cvec state, staterev, qustate;

qustate=itppextmath::BlochToQubit(theta.getValue(),phi.getValue());

//qustate=RandomState(64);

//int dim=pow_2(qubits.getValue());

qustate=BlochToQubit(theta.getValue(),phi.getValue());

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
	
if(option=="random")
	state=itppextmath::RandomState(pow(2,qubits.getValue()));

//cout<< qustate ;

staterev=state;

double Jrev=J.getValue()+Jpert.getValue();


if(option2=="fidelity"){

itpp::vec list(steps.getValue());

for(int i=0;i<steps.getValue();i++){

list(i)=pow( abs( dot( conj(staterev),state)),2);

//cout<< pow( abs( dot( conj(staterev),state)),2) <<endl;

cout << list(i) <<endl;
// cout<< i<< " " << list(i) <<endl;

list(i)=sqrt(list(i));

apply_ising_chain_inhom(state, J.getValue()+Jpert.getValue(), J.getValue()+Jinhompert.getValue()+Jpert.getValue());

apply_inhomogeneous_kick(state, bpert, binhom);

apply_ising_chain_inhom(staterev, J.getValue()-Jpert.getValue(), J.getValue()-Jinhompert.getValue()-Jpert.getValue());

apply_inhomogeneous_kick(staterev, bpertrev, binhomrev);

}
 
//fidelity.close();

//cout << staterev;

cout<< itppextmath::sum_positive_derivatives(list)<< endl;
}
//cout<<state<<endl;
if(option2=="correlacion"){
	
cvec list(steps.getValue());

cvec init=state;

for(int i=0;i<steps.getValue();i++){

list(i)=dot(conj(init),state);

cout << real(list(i)) << " " << imag(list(i)) <<endl;

//cout << list <<endl;

apply_ising_inhom(state, J.getValue(), J.getValue()+Jinhompert.getValue());

apply_magnetic_inhom(state, bpert, binhom);
}
}
if(option2=="fidelityandipr"){

vec listfidel(steps.getValue());

cvec listcorr(steps.getValue());

cvec init=state;

for(int i=0;i<steps.getValue();i++){

listfidel(i)=pow( abs( dot( conj(staterev),state)),2);

listcorr(i)=pow(abs(dot(conj(init),state)),2);

//cout<< pow( abs( dot( conj(staterev),state)),2) <<endl;

cout << listfidel(i) <<endl;
// cout<< i<< " " << list(i) <<endl;

listfidel(i)=sqrt(listfidel(i));

apply_ising_inhom(state, J.getValue()+Jpert.getValue(), J.getValue()+Jinhompert.getValue()+Jpert.getValue());

apply_magnetic_inhom(state, bpert, binhom);

apply_ising_inhom(staterev, J.getValue()-Jpert.getValue(), J.getValue()-Jinhompert.getValue()-Jpert.getValue());

apply_magnetic_inhom(staterev, bpertrev, binhomrev);

//cout<<abs(dot(conj(staterev),state))<<endl;

//fidelity<<pow(abs(dot(conj(staterev),state)),2)<<endl;

}
 
//fidelity.close();

//cout << staterev;

cout<< sum_positive_derivatives(listfidel)<< endl;

cout<< real(mean(listcorr))<< endl;
}

//cout<<binhom<<" "<<binhomrev<<endl;

}
