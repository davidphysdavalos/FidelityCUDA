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
//#include "tools.cpp"
#include "cuda_functions.cu"
#include "cuda_utils.cu"
#include "ev_routines.cu"
#include "cfp_routines.cu"

//~ using namespace std; 
//~ using namespace itpp;
//~ using namespace itppextmath;
//~ using namespace cfpmath;
//~ using namespace spinchain;
//~ using namespace RMT;
		
	void apply_ising_chain_inhom(itpp::cvec& state,double ising, double Jinhom) { // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		// cout << nqubits;
		int numthreads;
		int numblocks;
		choosenumblocks(l,numthreads,numblocks);
		//set_parameters(ising,b,icos,isin,kcos,ksin,bx,by,bz);

		itppcuda::itpp2cuda(state,&dev_R,&dev_I);


			for(int i=1;i<nqubits;i++) {
				Ui_kernel<<<numblocks,numthreads>>>(i,(i+1)%nqubits,dev_R,dev_I,cos(ising),sin(ising),l);
				//       cudaCheckError("ising",i);
			}
			Ui_kernel<<<numblocks,numthreads>>>(0,1,dev_R,dev_I,cos(Jinhom),sin(Jinhom),l);
		//cout<<dev_R<<" "<<dev_I<<endl;

		itppcuda::cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} // }}}
	
		void apply_inhomogeneous_kick(itpp::cvec& state ,itpp::vec b, itpp::vec binhom ){ // {{{
		double* dev_R;
		double* dev_I;
		int l=state.size();
		int nqubits=log(l)/log(2);
		// cout << nqubits;
		int numthreads;
		int numblocks;
		double theta=itpp::norm(b);
		double theta2=itpp::norm(binhom);
		choosenumblocks(l,numthreads,numblocks);

		itppcuda::itpp2cuda(state,&dev_R,&dev_I);
		
		Uk_kernel<<<numblocks,numthreads>>>(0,dev_R,dev_I,binhom(0)/theta2,binhom(1)/theta2,binhom(2)/theta2,cos(theta2),sin(theta2),l);

			for(int i=1;i<nqubits;i++) {
				Uk_kernel<<<numblocks,numthreads>>>(i,dev_R,dev_I,b(0)/theta,b(1)/theta,b(2)/theta,cos(theta),sin(theta),l);
				//       cudaCheckError("kick",i);
			}
		//cout<<dev_R<<" "<<dev_I<<endl;

		itppcuda::cuda2itpp(state,dev_R,dev_I);
		cudaFree(dev_R);
		cudaFree(dev_I);

	} 


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
TCLAP::ValueArg<double> deltabx("","deltabx", "perturbation campo en x",false, 0.0,"double",cmd);
TCLAP::ValueArg<double> deltabz("","deltabz", "perturbation campo en z",false, 0.0,"double",cmd);
TCLAP::ValueArg<int> steps("","steps","steps",false, 100,"int",cmd);
TCLAP::ValueArg<double> Jpert("","Jpert","Perturbation on Ising",false, 0.0,"double",cmd);
TCLAP::ValueArg<double> Jinhompert("","Jinhompert","Inhomogeneous perturbation on Ising on 0-1 interaction",false, 0.0,"double",cmd);
TCLAP::ValueArg<double> deltabxinhom("","deltabxinhom", "perturbation al campo solo en el qubit 0",false, 0.0,"double",cmd);
TCLAP::ValueArg<int> dev("","dev", "Gpu to be used, 0 for c20, 1 para la jodida",false, 0,"int",cmd);


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
// Para perturbacion en z
bpert(2)=b(2)+deltabz.getValue();


bpertrev(0)=b(0)-deltabx.getValue();
// Para perturbacion en z
bpertrev(2)=b(2)-deltabz.getValue();


binhom=bpert;
binhomrev=bpertrev;
binhom(0)=bpert(0)+deltabxinhom.getValue();
binhomrev(0)=bpertrev(0)-deltabxinhom.getValue();
string option=optionArg.getValue();
string option2=optionArg2.getValue();

itpp::cvec state, staterev, qustate;

qustate=itppextmath::BlochToQubit(theta.getValue(),phi.getValue());

{{{
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

}}}
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
if(option2=="fidelityandipr"){

itpp::vec listfidel(steps.getValue());

itpp::cvec listcorr(steps.getValue());

itpp::cvec init=state;

for(int i=0;i<steps.getValue();i++){

listfidel(i)=pow( abs( dot( conj(staterev),state)),2);

listcorr(i)=pow(abs(dot(conj(init),state)),2);

std::cout << listfidel(i) <<endl;

listfidel(i)=sqrt(listfidel(i));

apply_ising_chain_inhom(state, J.getValue()+Jpert.getValue(), J.getValue()+Jinhompert.getValue()+Jpert.getValue());

apply_inhomogeneous_kick(state, bpert, binhom);

apply_ising_chain_inhom(staterev, J.getValue()-Jpert.getValue(), J.getValue()-Jinhompert.getValue()-Jpert.getValue());

apply_inhomogeneous_kick(staterev, bpertrev, binhomrev);

}
 
//fidelity.close();

//cout << staterev;

cout<< itppextmath::sum_positive_derivatives(listfidel)<< endl;

cout<< real(mean(listcorr))<< endl;
}

if(option2=="correlationandipr"){

itpp::cvec listcorr(steps.getValue());

itpp::cvec init=state;

for(int i=0;i<steps.getValue();i++){

listcorr(i)=pow(abs(dot(conj(init),state)),2);

std::cout << listcorr(i) <<endl;

apply_ising_chain_inhom(state, J.getValue()+Jpert.getValue(), J.getValue()+Jinhompert.getValue()+Jpert.getValue());

apply_inhomogeneous_kick(state, bpert, binhom);

}

cout<< real(mean(listcorr))<< endl;
}

if(option2=="ipr"){

itpp::cvec listcorr(steps.getValue());

itpp::cvec init=state;

for(int i=0;i<steps.getValue();i++){

listcorr(i)=pow(abs(dot(conj(init),state)),2);

//std::cout << listcorr(i) <<endl;

apply_ising_chain_inhom(state, J.getValue()+Jpert.getValue(), J.getValue()+Jinhompert.getValue()+Jpert.getValue());

apply_inhomogeneous_kick(state, bpert, binhom);

}

cout<< real(mean(listcorr))<< endl;
}


}
