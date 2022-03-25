#define EIGEN_STACK_ALLOCATION_LIMIT 10000000000000
#define _USE_MATH_DEFINES

#include "reflectivity_functions_CPP.h"
//#include <iostream>
//#include <iomanip>
#include <cmath>
//#include <chrono>
#include <Eigen/StdVector>
//#include <boost/math/special_functions/bessel.hpp>
//#include <omp.h>
//#include <string>
//#include <sstream>
#include <cstdint>
//#include<omp.h>

using namespace std;
using namespace Eigen;


void compute_displ(Map<MatrixXd>& layers, Map<VectorXd>& freq, Map<VectorXcd>& source, Map<VectorXd>& receivers,
	Map<Vector4i>& options, Map<MatrixXcd>& output)
{
	/* Compute the u_z displacement vector using trapezoidal integration. */
	
	// Read in options
	bool direct = (options(0) == 1) ? true : false;  // Include the direct wave
	bool reverb = (options(1) == 1) ? true : false;  // Include reverberations
	bool f_window = (options(2) == 1) ? true : false;  // Apply frequency windowing
	bool velocity = (options(3) == 1) ? true : false;  // Output particle velocity
	bool acceleration = (options(3) == 2) ? true : false;  // Output particle acceleration
	// Default output is particle displacement
	
	// Layer properties
	VectorXd alphas = layers.col(0);
	VectorXd Qalphas = layers.col(1);
	VectorXd betas = layers.col(2);
	VectorXd Qbetas = layers.col(3);
	VectorXd rhos = layers.col(4);
	VectorXd d = layers.col(5);

	int nLayers = alphas.size();
	int nF = freq.size();
	int nRec = receivers.size();

	// Variable step size for two zones:
	double u1 = 0.;
	double u2 = 1.2 / betas.minCoeff();
	double u3 = 5. * u2;

	double du_s = 0.05 / (freq.maxCoeff() * receivers.maxCoeff());
	double du_l = 1. * du_s;

	int nu_s = round((u2 - u1) / du_s) + 500;
	int nu_l = round((u3 - u2) / du_l) + 500;
	int nU = nu_s + nu_l;

	ArrayXd u = ArrayXd::Zero(nU);
	u.head(nu_s) = ArrayXd::LinSpaced(nu_s, u1, u2);
	u.tail(nu_l) = ArrayXd::LinSpaced(nu_l, u2 + du_l, u3);
	Array<double, 1, Dynamic> du = u.tail(nU - 1) - u.head(nU - 1);  // du step vector (1xN vector for multiplying correctly later)
	
	// Create the slowness tapering window
	double u_taper = u2;  // start the tapering from this value onwards
	unsigned int t_idx = 0;
	for (int i = 0; i < u.size(); i++) {
		if (u(i) > u_taper) {
			t_idx = i;
			break;
		}
	}
	t_idx = nU - t_idx;
	ArrayXd slowness_window = ArrayXd::Ones(nU);
	slowness_window.tail(t_idx) = 0.5 * (1 + cos(M_PI * ((u.tail(t_idx) - u_taper) / (u(nU - 1) - u_taper))));

	MatrixXcd a = MatrixXcd::Zero(nU, nLayers);  // vertical P-wave slownesses of the layers
	MatrixXcd b = MatrixXcd::Zero(nU, nLayers);  // vertical S-wave slownesses of the layers

	double rho_m = rhos(0);

	double z_s = 0.;  // source depth

	// Precomputations -----------------------------------

	/* Approximate slownesses as independent of frequency, compute a single value
	using the dominant frequency of the problem. */
	// int freq_peak_idx = argmax <-- doesn't exist in Eigen...
	double w_dominant = 2. * M_PI * 80.;

	// Compute slownesses with attenuation
	VectorXcd uP = VectorXcd::Zero(nLayers);
	VectorXcd uS = VectorXcd::Zero(nLayers);
	VectorXcd vS = VectorXcd::Zero(nLayers);
	Q_slowness(alphas, Qalphas, betas, Qbetas, w_dominant, uP, uS, vS);

	vector<Matrix2cd, aligned_allocator<Matrix2cd> > Rplus(nU);  // Define a stl::vector of Eigen 2x2 matrices
	Rplus.reserve(nU);  // Reserve space for nU copies of the 2x2 matrix

	Matrix2Xcd H = Matrix2Xcd::Zero(2, nU);

	// Declare the most time-critical part (memory access wise) (see https://www.cplusplus.com/forum/articles/7459/)
	// I don't know if this is optimal or not
	Matrix2cd** Ru = {};  // Will be an array of pointers
	Matrix2cd** Tu = {};
	Matrix2cd** Rd = {};
	Matrix2cd** Td = {};

	if (nLayers > 1) {
		// Allocate memory
		Ru = new Matrix2cd * [nU];
		Tu = new Matrix2cd * [nU];
		Rd = new Matrix2cd * [nU];
		Td = new Matrix2cd * [nU];
		for (int i = 0; i < nU; i++)
		{
			Ru[i] = new Matrix2cd[nLayers - 1];
			Tu[i] = new Matrix2cd[nLayers - 1];
			Rd[i] = new Matrix2cd[nLayers - 1];
			Td[i] = new Matrix2cd[nLayers - 1];
		}
	}

	// Compute vertical slownesses of layers 1 to n for every u
	for (int kk = 0; kk < nLayers; kk++)
	{
		a.col(kk) = sqrt(pow(uP(kk), 2) - pow(u, 2));
		b.col(kk) = sqrt(pow(uS(kk), 2) - pow(u, 2));
	}

	complex<double> Hcoeff;
	Vector2cd Hvect;
	// Note: parallelizing this loop only made the program slower
	for (int ii = 0; ii < nU; ii++)
	{
		// Free surface
		Rplus[ii] = Rplus_freesurface(a(ii, 0), b(ii, 0), rhos(0), vS(0), u(ii));

		Hcoeff = 1. / (pow(1. - 2. * pow(vS(0), 2) * pow(u(ii), 2), 2) + 4. * pow(vS(0), 4) * pow(u(ii), 2) * a(ii, 0) * b(ii, 0));
		Hvect = Vector2cd((1. - 2. * pow(vS(0), 2) * pow(u(ii), 2)) * a(ii, 0),
			-2. * pow(vS(0), 2) * u(ii) * a(ii, 0) * b(ii, 0));
		H.col(ii) = Hcoeff * Hvect;

		// Precompute stuff for Rminus
		// start with ii = n - 1
		for (int kk = nLayers - 1 - 1; kk > -1; kk--)
		{
			computeRT(a(ii, kk), b(ii, kk), rhos(kk), vS(kk), a(ii, kk + 1), b(ii, kk + 1), rhos(kk + 1), vS(kk + 1), u(ii),
				Ru[ii][kk], Tu[ii][kk], Rd[ii][kk], Td[ii][kk]);
		}
	}

	// Start the main loop ---------------------------------------------
	/* The static scheduling with chunk size 1 works quite well in balancing the load
	between threads with the trapezoidal integration. It is much more efficient than
	the default schedule (in the cases I've tested). I haven't seen an improvement with
	Levin integration cases. */
#pragma omp parallel for schedule(static, 1)
	for (int jj = 0; jj < nF; jj++)
	{
		// For thread saefty, these need to be declared inside the parallelized loop
		MatrixXcd integrand = MatrixXd::Zero(nRec, nU);
		Matrix2cd Rminus = Matrix2cd::Zero(2, 2);
		Vector2cd V1;
		
		Vector2cd S_1u, S_1d;
		ArrayXcd J1 = ArrayXcd::Zero(nRec);
		
		double omega = 2. * M_PI * freq(jj);
		for (int ii = 0; ii < nU; ii++)
		{
			// source terms
			complex<double> a_m = a(ii, 0);
			complex<double> b_m = b(ii, 0);
			complex<double> e_alpha = exp(1i * omega * a_m * (z_s - 0));
			complex<double> e_beta = exp(1i * omega * b_m * (z_s - 0));

			S_1u << -u(ii) * pow(e_alpha, -1),
				pow(u(ii), 2) / b_m * pow(e_beta, -1);
			S_1d << u(ii) * e_alpha,
				pow(u(ii), 2) / b_m * e_beta;

			// reflectivity
			VectorXcd row_of_a = a.row(ii);
			VectorXcd row_of_b = b.row(ii);

			if (nLayers > 1) {
				computeRminus(Ru[ii], Tu[ii], Rd[ii], Td[ii], nLayers, row_of_a, row_of_b, d, omega, Rminus);
			}

			if (!direct) {
				if (!reverb) {
					V1 = Rminus * S_1d;  // no direct nor reverberations
				}
				else {
					V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (Rminus * S_1d); // no direct wave
				}
			}
			else if (!reverb) {
				V1 = S_1u + Rminus * S_1d;  // no reverberations
			}
			else {
				V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (S_1u + Rminus * S_1d);
			}

			for (int rec = 0; rec < nRec; rec++)
			{
				J1(rec) = 1i * std::cyl_bessel_j(0, u(ii) * omega * receivers(rec));
			}
			integrand.block(0, ii, nRec, 1) = 2. * J1 * (H(0, ii) * V1(0) + H(1, ii) * V1(1));
		}

		// Multiply each row by the slowness window
		for (int i = 0; i < nRec; i++)
		{
			integrand.block(i, 0, 1, nU) = integrand.block(i, 0, 1, nU).array() * slowness_window.transpose();
		}

		for (int rec = 0; rec < nRec; rec++)
		{
			// Trapezoidal integration with constant stepsize
			//output(jj, rec) = du * (integrand.block(rec, 1, 1, nU - 2).sum() + 0.5 * (integrand(rec, 0) + integrand(rec, nU - 1)));

			// Trapezoidal integration with a variable stepsize
			output(jj, rec) = 0.5 * ((integrand.block(rec, 0, 1, nU - 1) + integrand.block(rec, 1, 1, nU - 1)).array() * du).sum();

			// Scaling
			output(jj, rec) *= omega * source(jj) / (4. * M_PI * rho_m);
		}
	}
	
	// Change to velocity or acceleration
	if (velocity) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * 1i * 2. * M_PI * freq.array();
		}
	}
	else if (acceleration) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * -4. * M_PI * M_PI * freq.array() * freq.array();
		}
	}
	
	if (nLayers > 1) {
		// Deallocate dynamic memory
		for (int ii = 0; ii < nU; ii++)
		{
			delete[] Ru[ii];
			delete[] Tu[ii];
			delete[] Rd[ii];
			delete[] Td[ii];
		}
		delete[] Ru;
		delete[] Tu;
		delete[] Rd;
		delete[] Td;
	}
	
	// Frequency windowing
	if (f_window) {
		ArrayXd freq_window = tukey(nF, 0.7);
		for (int rec = 0; rec < nRec; rec++)
		{
			output.col(rec) = output.col(rec).array() * freq_window;
		}
	}
}


void computeRminus(Matrix2cd* Ru, Matrix2cd* Tu, Matrix2cd* Rd, Matrix2cd* Td, int nLayers,
	VectorXcd& a, VectorXcd& b, VectorXd& d, double omega, Matrix2cd& Rminus)
{
	int startlayer = nLayers - 1 - 1;

	Matrix2cd MT = Matrix2cd::Zero();  // Start with MT_n == 0
	Matrix2cd MB = Rd[startlayer];
	Matrix2cd E;
	complex<double> l, lp;

	// First iteration
	l = omega * a(startlayer);
	lp = omega * b(startlayer);
	E(0, 0) = exp(-1i * 2. * l * d(startlayer));
	E(0, 1) = exp(-1i * (l + lp) * d(startlayer));
	E(1, 0) = E(0, 1);
	E(1, 1) = exp(-1i * 2. * lp * d(startlayer));
	MT = MB.array() * E.array();
	
	for (int ii = startlayer - 1; ii > -1; ii--)
	{
		MB = Rd[ii] + Tu[ii] * (Matrix2d::Identity() - MT * Ru[ii]).inverse() * MT * Td[ii];

		l = omega * a(ii);
		lp = omega * b(ii);

		E(0, 0) = exp(-1i * 2. * l * d(ii));
		E(0, 1) = exp(-1i * (l + lp) * d(ii));
		E(1, 0) = E(0, 1);
		E(1, 1) = exp(-1i * 2. * lp * d(ii));

		// This should be element-wise
		MT = MB.array() * E.array();
	}

	Rminus = MT;
}


void computeRT(complex<double> a1, complex<double> b1, double rho1, complex<double> vS1, complex<double> a2, complex<double> b2,
	double rho2, complex<double> vS2, double u, Matrix2cd& Ru, Matrix2cd& Tu, Matrix2cd& Rd, Matrix2cd& Td)
{
	complex<double> mu1 = rho1 * pow(vS1, 2);
	complex<double> mu2 = rho2 * pow(vS2, 2);
	complex<double> c = 2. * (mu1 - mu2);

	complex<double> uto2 = pow(u, 2);  // to speed things up...
	complex<double> cu2 = c * uto2;

	// upgoing
	complex<double> D1u = pow(cu2 - rho1 + rho2, 2) * uto2 + pow(cu2 + rho2, 2) * a1 * b1 + rho1 * rho2 * a1 * b2;
	complex<double> D2u = pow(c, 2) * uto2 * a1 * a2 * b1 * b2 + pow(cu2 - rho1, 2) * a2 * b2 + rho1 * rho2 * a2 * b1;

	complex<double> invD12u = 1. / (D1u + D2u);  // to speed things up...

	complex<double> Tupp = 2. * rho2 * a2 * invD12u * ((cu2 + rho2) * b1 - (cu2 - rho1) * b2);
	complex<double> Tups = -2. * rho2 * u * a2 * invD12u * (cu2 - rho1 + rho2 + c * a1 * b2);
	complex<double> Tusp = 2. * rho2 * u * b2 * invD12u * (cu2 - rho1 + rho2 + c * a2 * b1);
	complex<double> Tuss = 2. * rho2 * b2 * invD12u * ((cu2 + rho2) * a1 - (cu2 - rho1) * a2);

	Tu << Tupp, Tusp,
		Tups, Tuss;

	complex<double> Rupp = (D2u - D1u) * invD12u;
	complex<double> Rups = 2. * u * a2 * invD12u * ((cu2 - rho1 + rho2) * (cu2 - rho1) + c * (cu2 + rho2) * a1 * b1);
	complex<double> Rusp = -2. * u * b2 * invD12u * ((cu2 - rho1 + rho2) * (cu2 - rho1) + c * (cu2 + rho2) * a1 * b1);
	complex<double> Russ = (D2u - D1u - 2. * rho1 * rho2 * (a2 * b1 - a1 * b2)) * invD12u;

	Ru << Rupp, Rusp,
		Rups, Russ;

	// downgoing
	complex<double> D1d = pow(cu2 - rho1 + rho2, 2) * uto2 + pow(cu2 - rho1, 2) * a2 * b2 + rho1 * rho2 * a2 * b1;
	complex<double> D2d = pow(c, 2) * uto2 * a1 * a2 * b1 * b2 + pow(cu2 + rho2, 2) * a1 * b1 + rho1 * rho2 * a1 * b2;

	complex<double> invD12d = 1. / (D1d + D2d);  // to speed things up...

	complex<double> Rdpp = (D2d - D1d) * invD12d;
	complex<double> Rdps = -2. * u * a1 * invD12d * ((cu2 - rho1 + rho2) * (cu2 + rho2) + c * (cu2 - rho1) * a2 * b2);
	complex<double> Rdsp = 2. * u * b1 * invD12d * ((cu2 - rho1 + rho2) * (cu2 + rho2) + c * (cu2 - rho1) * a2 * b2);
	complex<double> Rdss = (D2d - D1d - 2. * rho1 * rho2 * (a1 * b2 - a2 * b1)) * invD12d;

	Rd << Rdpp, Rdsp,
		Rdps, Rdss;

	complex<double> Tdpp = 2. * rho1 * a1 * invD12d * ((cu2 + rho2) * b1 - (cu2 - rho1) * b2);
	complex<double> Tdps = -2. * rho1 * u * a1 * invD12d * (cu2 - rho1 + rho2 + c * a2 * b1);
	complex<double> Tdsp = 2. * rho1 * u * b1 * invD12d * (cu2 - rho1 + rho2 + c * a1 * b2);
	complex<double> Tdss = 2. * rho1 * b1 * invD12d * ((cu2 + rho2) * a1 - (cu2 - rho1) * a2);

	Td << Tdpp, Tdsp,
		Tdps, Tdss;
}


void Q_slowness(VectorXd& vP, VectorXd& QP, VectorXd& vS, VectorXd& QS, double w, VectorXcd& uP, VectorXcd& uS, VectorXcd& vSQ)
{
/* Computes the complex, frequency-dependent, slownesses for P- and S-waves based on the P- and S-wave
speeds and a Q-model. See Müller (1985), eq. 132.

	Input
    ----------
    vP : P-wave speed (can be a vector)
    QP : P-wave Q-factor (can be a vector)
    vS : S-wave speed (can be a vector)
    QS : S-wave Q-factor (can be a vector)
    w : circular frequency (omega) NOTE: for computational efficiency reasons, we assume that attenuation
		is independent of frequency, and compute the attenuation coefficients for a dominant frequency
		only so that the interface reflection and transmission coefficients continue to depend only on slowness.
		This causes the absorption to become slightly acausal.
	    

    Output
    -------
    uP : P-wave slownesses
    uS : S-wave slownesses
    vS : complex S-wave speeds */

	double w_ref = 2. * M_PI * 80.;  // Q reference frequency. If this is the same as the dominant frequency w,
									 // they cancel each other out, and we don't need to in principle ever change
									 // these numbers even if the frequency content of the source changes.
	uP = 1. / (vP.array() * (1. + 1. / (M_PI * QP.array()) * log(w / w_ref) + 1i / (2. * QP.array())));
	vSQ = vS.array() * (1. + 1. / (M_PI * QS.array()) * log(w / w_ref) + 1i / (2. * QS.array()));
	uS = 1. / vSQ.array();
}

Matrix2cd Rplus_freesurface(complex<double> a2, complex<double> b2, double rho2, complex<double> vS2, double u)
{
	/* Computes the (single layer) matrices of interface reflection and
	transmission coefficients for downgoing waves.
	
	Returns: Rplus for the free surface (source in 1st layer) */

	complex<double> Sslow = 1. / vS2;  // slowness of S-wave
	complex<double> T1 = 4. * a2 * b2 * pow(u, 2);
	complex<double> T2_sqrt = 2. * pow(u, 2) - pow(Sslow, 2);
	complex<double> T2 = pow(T2_sqrt, 2);

	complex<double> Rdpp = (T1 - T2) / (T1 + T2);
	complex<double> Rdps = 4. * u * a2 * T2_sqrt / (T1 + T2);
	complex<double> Rdsp = -4. * u * b2 * T2_sqrt / (T1 + T2);
	complex<double> Rdss = (T1 - T2) / (T1 + T2);

	Matrix2cd Rd;
	Rd << Rdpp, Rdsp,
		  Rdps, Rdss;

	return Rd;
}


ArrayXd tukey(int length, double alpha)
{
	if (alpha <= 0 || alpha >= 1) { return ArrayXd::Ones(length); }

	// This function looks suspiciously like scipy.signal.tukey ;)

	ArrayXd n = ArrayXd::LinSpaced(length, 0, length - 1);
	int width = (int)floor(alpha * (length - 1) / 2.0);

	ArrayXd n1 = n.head(width + 1);
	ArrayXd n2 = n.segment(width + 1, length - 2 * width - 2);
	ArrayXd n3 = n.tail(width + 1);

	ArrayXd w1 = 0.5 * (1 + cos(M_PI * (-1. + 2.0 * n1 / alpha / (length - 1))));
	ArrayXd w2 = ArrayXd::Ones(n2.size());
	ArrayXd w3 = 0.5 * (1 + cos(M_PI * (-2.0 / alpha + 1. + 2.0 * n3 / alpha / (length - 1))));

	ArrayXd window(w1.size() + w2.size() + w3.size());
	window << w1, w2, w3;

	return window;
}


void compute_displ_Levin(Map<MatrixXd>& layers, Map<VectorXd>& freq, Map<VectorXcd>& source, Map<VectorXd>& receivers,
	Map<Vector4i>& options, Map<MatrixXcd>& output)
{
	/* Compute the u_z displacement vector using Levin integration. */

	// Read in options
	bool direct = (options(0) == 1) ? true : false;  // Include the direct wave
	bool reverb = (options(1) == 1) ? true : false;  // Include reverberations
	bool f_window = (options(2) == 1) ? true : false;  // Apply frequency windowing
	bool velocity = (options(3) == 1) ? true : false;  // Output particle velocity
	bool acceleration = (options(3) == 2) ? true : false;  // Output particle acceleration
	// Default output is particle displacement

	// Layer properties
	VectorXd alphas = layers.col(0);
	VectorXd Qalphas = layers.col(1);
	VectorXd betas = layers.col(2);
	VectorXd Qbetas = layers.col(3);
	VectorXd rhos = layers.col(4);
	VectorXd d = layers.col(5);

	int nLayers = alphas.size();
	int nF = freq.size();
	int nRec = receivers.size();

	// Two subinterval 'zones':
	double u1 = 1e-3 / alphas.maxCoeff();
	double u2 = 1.2 / betas.minCoeff();
	double u3 = 5. * u2;

	// Set the number of collocation points per subinterval
	const int n = 12;

	// Compute how many subintervals Q do we need (this is based on trial and error...)
	double du = 0.001 / sqrt(freq.maxCoeff() * receivers.maxCoeff());
	double min_fevals_primary_interval = (u2 - u1) / du + 1000.;
	double min_fevals_secondary_interval = 100.;

	int Q_primary = (int)ceil(min_fevals_primary_interval / (n - 1));
	int Q_secondary = (int)ceil(min_fevals_secondary_interval / (n - 1));
	int Q = Q_primary + Q_secondary;

	// Divide the integration path into Q subintervals with n collocation points each (including endpoints)
	// Use Chebyshev points

	ArrayXd u = ArrayXd::Zero((n - 1) * Q + 1);
	ArrayXd u_subintv = ArrayXd::Zero(Q + 1);
	u_subintv.head(Q_primary + 1) = ArrayXd::LinSpaced(Q_primary + 1, u1, u2);
	u_subintv.tail(Q_secondary + 1) = ArrayXd::LinSpaced(Q_secondary + 1, u2, u3);
	for (int q = 0; q < Q; q++)
	{
		u.segment<n>(q * n - q) = (0.5 * (u_subintv(q) + u_subintv(q + 1))
			+ 0.5 * (u_subintv(q + 1) - u_subintv(q))
			* cos((2. * ArrayXd::LinSpaced(n, 0., n - 1)) / (2 * (n - 1)) * M_PI)).reverse();
	}

	int nU = u.size();

	// Create the slowness tapering window
	double u_taper = u2;  // start the tapering from this value onwards
	unsigned int t_idx = 0;
	for (int i = 0; i < u.size(); i++) {
		if (u(i) > u_taper) {
			t_idx = i;
			break;
		}
	}
	t_idx = nU - t_idx;
	ArrayXd slowness_window = ArrayXd::Ones(nU);
	slowness_window.tail(t_idx) = 0.5 * (1 + cos(M_PI * ((u.tail(t_idx) - u_taper) / (u(nU - 1) - u_taper))));

	MatrixXcd a = MatrixXcd::Zero(nU, nLayers);  // vertical P-wave slownesses of the layers
	MatrixXcd b = MatrixXcd::Zero(nU, nLayers);  // vertical S-wave slownesses of the layers

	double rho_m = rhos(0);
	double z_s = 0.;  // source depth

	// Precomputations -----------------------------------

	/* Approximate slownesses as independent of frequency, compute a single value
	using the dominant frequency of the problem. */
	// int freq_peak_idx = argmax <-- doesn't exist in Eigen...
	double w_dominant = 2. * M_PI * 80.;

	// Compute slownesses with attenuation
	VectorXcd uP = VectorXcd::Zero(nLayers);
	VectorXcd uS = VectorXcd::Zero(nLayers);
	VectorXcd vS = VectorXcd::Zero(nLayers);
	Q_slowness(alphas, Qalphas, betas, Qbetas, w_dominant, uP, uS, vS);

	vector<Matrix2cd, aligned_allocator<Matrix2cd> > Rplus(nU);  // Define a stl::vector of Eigen 2x2 matrices
	Rplus.reserve(nU);  // Reserve space for nU copies of the 2x2 matrix

	Matrix2Xcd H = Matrix2Xcd::Zero(2, nU);

	// Declare the most time-critical part (memory access wise) (see https://www.cplusplus.com/forum/articles/7459/)
	// I don't know if this is optimal or not
	Matrix2cd** Ru = {};  // Will be an array of pointers
	Matrix2cd** Tu = {};
	Matrix2cd** Rd = {};
	Matrix2cd** Td = {};

	if (nLayers > 1) {
		// Allocate memory
		Ru = new Matrix2cd * [nU];
		Tu = new Matrix2cd * [nU];
		Rd = new Matrix2cd * [nU];
		Td = new Matrix2cd * [nU];
		for (int ii = 0; ii < nU; ii++)
		{
			Ru[ii] = new Matrix2cd[nLayers - 1];
			Tu[ii] = new Matrix2cd[nLayers - 1];
			Rd[ii] = new Matrix2cd[nLayers - 1];
			Td[ii] = new Matrix2cd[nLayers - 1];
		}
	}

	// Compute vertical slownesses of layers 1 to n for every u
	for (int kk = 0; kk < nLayers; kk++)
	{
		a.col(kk) = sqrt(pow(uP(kk), 2) - pow(u, 2));
		b.col(kk) = sqrt(pow(uS(kk), 2) - pow(u, 2));
	}

	complex<double> Hcoeff;
	Vector2cd Hvect;
	for (int ii = 0; ii < nU; ii++)
	{
		// Free surface
		Rplus[ii] = Rplus_freesurface(a(ii, 0), b(ii, 0), rhos(0), vS(0), u(ii));

		Hcoeff = 1. / (pow(1. - 2. * pow(vS(0), 2) * pow(u(ii), 2), 2) + 4. * pow(vS(0), 4) * pow(u(ii), 2) * a(ii, 0) * b(ii, 0));
		Hvect = Vector2cd((1. - 2. * pow(vS(0), 2) * pow(u(ii), 2)) * a(ii, 0),
			-2. * pow(vS(0), 2) * u(ii) * a(ii, 0) * b(ii, 0));
		H.col(ii) = Hcoeff * Hvect;

		// Precompute stuff for Rminus
		// start with ii = n - 1
		for (int kk = nLayers - 1 - 1; kk > -1; kk--)
		{
			computeRT(a(ii, kk), b(ii, kk), rhos(kk), vS(kk), a(ii, kk + 1), b(ii, kk + 1), rhos(kk + 1), vS(kk + 1), u(ii),
				Ru[ii][kk], Tu[ii][kk], Rd[ii][kk], Td[ii][kk]);
		}
	}

	// Compute the basis functions for Levin integration
	vector<Matrix<double, 2*n, 2*n>, aligned_allocator<Matrix<double, 2*n, 2*n>> > basismat(Q);
	basismat.reserve(Q);  // not sure if this does anything...
	
	Matrix2Xd basis_u_interval = Matrix2Xd::Zero(2, n);
	MatrixXd basis;
	MatrixXd basisp;

	for (int q = 0; q < Q; q++)
	{
		Array<double, n, 1> u_interval = u.segment<n>(q * n - q);
		if (u_interval(0) == 0.) { u_interval(0) = 1e-12; }  // to avoid dividing by zero
		//basisfunctions(u_interval, basis, basisp);
		basisfunctions_radial(u_interval, basis, basisp);
		basismat[q].topLeftCorner(n, n) = basisp;
		basismat[q].topRightCorner(n, n) = basis;
		basismat[q].bottomLeftCorner(n, n) = -basis;
		basismat[q].bottomRightCorner(n, n) = basisp - (basis.array().colwise() / u_interval).matrix();
	}
	basis_u_interval.row(0) = basis.row(0);
	basis_u_interval.row(1) = basis.row(n - 1);
	
	// Start the main loop ---------------------------------------------

	double r_old = 1.;
	double practically_zero = 1e-30;
	int start_iter = 0;
	if (freq(0) <= practically_zero) {
		start_iter = 1;
		output.row(0) = VectorXcd::Zero(nRec);
	}

#pragma omp parallel for firstprivate(r_old, basismat)
	for (int jj = start_iter; jj < nF; jj++)
	{
		VectorXcd fpart = VectorXd::Zero(2 * nU);
		Matrix2cd Rminus = Matrix2cd::Zero(2, 2);
		Vector2cd V1;

		Vector2cd S_1u, S_1d;
		double omega = 2. * M_PI * freq(jj);
		for (int ii = 0; ii < nU; ii++)
		{
			// source terms
			complex<double> a_m = a(ii, 0);
			complex<double> b_m = b(ii, 0);
			complex<double> e_alpha = exp(1i * omega * a_m * (z_s - 0));
			complex<double> e_beta = exp(1i * omega * b_m * (z_s - 0));

			S_1u << -u(ii) * pow(e_alpha, -1),
				pow(u(ii), 2) / b_m * pow(e_beta, -1);
			S_1d << u(ii) * e_alpha,
				pow(u(ii), 2) / b_m * e_beta;

			// reflectivity
			VectorXcd row_of_a = a.row(ii);
			VectorXcd row_of_b = b.row(ii);

			if (nLayers > 1) {
				computeRminus(Ru[ii], Tu[ii], Rd[ii], Td[ii], nLayers, row_of_a, row_of_b, d, omega, Rminus);
			}

			if (!direct) {
				if (!reverb) {
					V1 = Rminus * S_1d;  // no direct nor reverberations
				}
				else {
					V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (Rminus * S_1d); // no direct wave
				}
			}
			else if (!reverb) {
				V1 = S_1u + Rminus * S_1d;  // no reverberations
			}
			else {
				V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (S_1u + Rminus * S_1d);
			}
			fpart(ii) = 1i * 2. * (H(0, ii) * V1(0) + H(1, ii) * V1(1));
		}

		// Taper by the slowness window
		fpart.head(nU) = fpart.head(nU).array() * slowness_window;

		VectorXcd rhs = VectorXcd::Zero(2 * n);
		for (int rec = 0; rec < nRec; rec++)
		{
			// Levin integration, compute the integral in Q parts
			double r = omega * receivers(rec);
			double r_change = r / r_old;
			r_old = r;

			ArrayXd bessel_J0 = ArrayXd::Zero(Q + 1);
			ArrayXd bessel_J1 = ArrayXd::Zero(Q + 1);
			for (int q = 0; q < Q; q++) {
				bessel_J0(q) = std::cyl_bessel_j(0, r * u_subintv(q));
				bessel_J1(q) = std::cyl_bessel_j(1, r * u_subintv(q));
			}

			complex<double> integral = 0.5 * u1 * bessel_J0(0) * fpart(0);

			for (int q = 0; q < Q; q++)
			{
				basismat[q].topRightCorner(n, n) *= r_change;
				basismat[q].bottomLeftCorner(n, n) *= r_change;
				rhs.head(n) = fpart.segment(q * n - q, n);

				VectorXcd c = basismat[q].lu().solve(rhs);

				integral += ((c.head(n).array() * basis_u_interval.row(1).array().transpose()).sum() * bessel_J0(q + 1)
					- (c.head(n).array() * basis_u_interval.row(0).array().transpose()).sum() * bessel_J0(q)
					+ (c.tail(n).array() * basis_u_interval.row(1).array().transpose()).sum() * bessel_J1(q + 1)
					- (c.tail(n).array() * basis_u_interval.row(0).array().transpose()).sum() * bessel_J1(q));
			}

			// Scaling
			output(jj, rec) = omega * source(jj) * integral / (4. * M_PI * rho_m);
		}
	}

	// Change to velocity or acceleration
	if (velocity) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * 1i * 2. * M_PI * freq.array();
		}
	}
	else if (acceleration) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * -4. * M_PI * M_PI * freq.array() * freq.array();
		}
	}
	
	if (nLayers > 1) {
		// Deallocate dynamic memory
		for (int ii = 0; ii < nU; ii++)
		{
			delete[] Ru[ii];
			delete[] Tu[ii];
			delete[] Rd[ii];
			delete[] Td[ii];
		}
		delete[] Ru;
		delete[] Tu;
		delete[] Rd;
		delete[] Td;
	}
	
	// Frequency windowing
	if (f_window) {
		ArrayXd freq_window = tukey(nF, 0.7);
		for (int rec = 0; rec < nRec; rec++)
		{
			output.col(rec) = output.col(rec).array() * freq_window;
		}
	}
}

void basisfunctions(ArrayXd u, MatrixXd& basis_out, MatrixXd& basisp_out)
{
	int n = u.size();  // this will tell us the size of the basis and basisp matrices too
	double d = u(0) + 0.5 * (u(n - 1) - u(0));  // midpoint
	double d0 = abs(u - d).maxCoeff();  // to normalise bases if wanted

	MatrixXd basis = MatrixXd::Ones(n, n);
	MatrixXd basisp = MatrixXd::Zero(n, n);
	for (int k = 1; k < n; k++)
	{
		basis.col(k) = pow((u - d) / d0, (double)k);
		basisp.col(k) = (double)k / d0 * pow((u - d) / d0, (double)k - 1.);
	}
	basis_out = basis;
	basisp_out = basisp;
}


void basisfunctions_radial(ArrayXd u, MatrixXd& basis_out, MatrixXd& basisp_out)
{
	int n = u.size();  // this will tell us the size of the basis and basisp matrices too
	double d = u(0) + 0.5 * (u(n - 1) - u(0));  // midpoint
	double d0 = abs(u - d).maxCoeff();
	double eps = 1 / d0;
	ArrayXd centerpoints = u - 0.25 * (u(n - 1) - u(0));
	MatrixXd basis = MatrixXd::Zero(n, n);
	MatrixXd basisp = MatrixXd::Zero(n, n);
	ArrayXd r = ArrayXd::Zero(n);
	for (int k = 0; k < n; k++)
	{
		r = u - centerpoints(k);
		basis.col(k) = sqrt(1 + pow(eps, 2) * pow(r, 2));
		basisp.col(k) = pow(eps, 2) * r / basis.col(k).array();
	}
	basis_out = basis;
	basisp_out = basisp;
}


uintptr_t precompute_Levin_basis(Map<VectorXd>& freq, Map<VectorXd>& receivers, Map<VectorXd>& u, int n_colloc)
{
	/* Generate a 3D array (frequency, receivers, slownesses) of pointers to a vector that is the product
	* of the Bessel functions at the ends of the interval, a matrix of basis functions at the ends of the
	* interval, and the inverse of the basis functions matrix. Return a pointer to the beginning of this
	* array, cast as an integer type so that it can be passed around in Python.
	* To make usage easier, I decided to get rid of the requirement to know n_colloc beforehand. This
	* means that we won't be able to take advantage of knowing the matrix dimensions during compilation,
	* but since this function is only run once per MCMC run it doesn't matter.
	*/

	int nF = freq.size();
	int nRec = receivers.size();
	int nU = u.size();
	int Q = (nU - 1) / (n_colloc - 1);

	ArrayXd u_subintv = ArrayXd::Zero(Q + 1);
	u_subintv = u(seq(0, nU, n_colloc - 1));

	vector<MatrixXd, aligned_allocator<MatrixXd> > basismat(Q);
	basismat.reserve(Q);  // not sure if this does anything...

	VectorXd*** Levin_basis = new VectorXd**[nF];
	for (int jj = 0; jj < nF; jj++) {
		Levin_basis[jj] = new VectorXd*[nRec];
		for (int rec = 0; rec < nRec; rec++) {
			Levin_basis[jj][rec] = new VectorXd[Q];
		}
	}

#pragma omp parallel for firstprivate(basismat)
	for (int q = 0; q < Q; q++)
	{
		MatrixXd basis;
		MatrixXd basisp;
		Vector4d bessels;
		Matrix4Xd U_basis = Matrix4Xd::Zero(4, 2 * n_colloc);

		ArrayXd u_interval = u.segment(q * n_colloc - q, n_colloc);
		
		basisfunctions_radial(u_interval, basis, basisp);
		basismat[q] = MatrixXd::Zero(2 * n_colloc, 2 * n_colloc);
		
		basismat[q].topLeftCorner(n_colloc, n_colloc) = basisp;
		basismat[q].topRightCorner(n_colloc, n_colloc) = basis;
		basismat[q].bottomLeftCorner(n_colloc, n_colloc) = -basis;
		basismat[q].bottomRightCorner(n_colloc, n_colloc) = basisp - (basis.array().colwise() / u_interval).matrix();

		U_basis.block(0, 0, 1, n_colloc) = basis.row(n_colloc - 1);
		U_basis.block(1, 0, 1, n_colloc) = basis.row(0);
		U_basis.block(2, n_colloc, 1, n_colloc) = basis.row(n_colloc - 1);
		U_basis.block(3, n_colloc, 1, n_colloc) = basis.row(0);

		double r_old = 1.;
		for (int jj = 0; jj < nF; jj++) {
			for (int rec = 0; rec < nRec; rec++) {
				double r = 2 * M_PI * freq[jj] * receivers[rec];
				bessels(0) = std::cyl_bessel_j(0, r * u_subintv(q + 1));
				bessels(1) = -std::cyl_bessel_j(0, r * u_subintv(q));
				bessels(2) = std::cyl_bessel_j(1, r * u_subintv(q + 1));
				bessels(3) = -std::cyl_bessel_j(1, r * u_subintv(q));
				
				if (jj == 0) { r = 1.; }
				double r_change = r / r_old;
				r_old = r;
				basismat[q].topRightCorner(n_colloc, n_colloc) *= r_change;
				basismat[q].bottomLeftCorner(n_colloc, n_colloc) *= r_change;

				Levin_basis[jj][rec][q] = bessels.transpose() * U_basis * basismat[q].inverse();
			}
		}
	}
	uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(Levin_basis);

	return ptr_as_int;
}


void deallocate_Levin_basis(uintptr_t Levin_basis_address, int nF, int nRec)
{
	VectorXd*** Levin_basis = reinterpret_cast<VectorXd***>(Levin_basis_address);

	for (int jj = 0; jj < nF; jj++) {
		for (int rec = 0; rec < nRec; rec++) {
			delete[] Levin_basis[jj][rec];
		}
		delete[] Levin_basis[jj];
	}
	delete[] Levin_basis;
}


void compute_displ_Levin_precomp(uintptr_t Levin_basis_address, Map<ArrayXd>& u, Map<ArrayXd>& slowness_window, int n,
	Map<MatrixXd>& layers, Map<VectorXd>& freq, Map<VectorXcd>& source, Map<VectorXd>& receivers, Map<Vector4i>& options, Map<MatrixXcd>& output)
{
	/* Compute the u_z displacement vector using Levin integration and precomputed vectors that include the Bessel functions and Levin
	basis matrices. */

	// Read in options
	bool direct = (options(0) == 1) ? true : false;  // Include the direct wave
	bool reverb = (options(1) == 1) ? true : false;  // Include reverberations
	bool f_window = (options(2) == 1) ? true : false;  // Apply frequency windowing
	bool velocity = (options(3) == 1) ? true : false;  // Output particle velocity
	bool acceleration = (options(3) == 2) ? true : false;  // Output particle acceleration
	// Default output is particle displacement

	// Layer properties
	VectorXd alphas = layers.col(0);
	VectorXd Qalphas = layers.col(1);
	VectorXd betas = layers.col(2);
	VectorXd Qbetas = layers.col(3);
	VectorXd rhos = layers.col(4);
	VectorXd d = layers.col(5);

	int nLayers = alphas.size();
	int nF = freq.size();
	int nRec = receivers.size();
	int nU = u.size();
	int Q = (nU - 1) / (n - 1);

	ArrayXd u_subintv = ArrayXd::Zero(Q + 1);
	u_subintv = u(seq(0, nU, n - 1));
	
	MatrixXcd a = MatrixXcd::Zero(nU, nLayers);  // vertical P-wave slownesses of the layers
	MatrixXcd b = MatrixXcd::Zero(nU, nLayers);  // vertical S-wave slownesses of the layers

	double rho_m = rhos(0);
	double z_s = 0.;  // source depth

	// Check if we want to compute an impulse response
	VectorXcd sourcefunction;
	if (source.size() == 1) {
		sourcefunction = VectorXcd::Ones(nF);
	}
	else {
		sourcefunction = source;
	}

	// Precomputations -----------------------------------

	/* Approximate slownesses as independent of frequency, compute a single value
	using the dominant frequency of the problem. */
	// int freq_peak_idx = argmax <-- doesn't exist in Eigen...
	double w_dominant = 2. * M_PI * 80.;

	// Compute slownesses with attenuation
	VectorXcd uP = VectorXcd::Zero(nLayers);
	VectorXcd uS = VectorXcd::Zero(nLayers);
	VectorXcd vS = VectorXcd::Zero(nLayers);
	Q_slowness(alphas, Qalphas, betas, Qbetas, w_dominant, uP, uS, vS);

	vector<Matrix2cd, aligned_allocator<Matrix2cd> > Rplus(nU);  // Define a stl::vector of Eigen 2x2 matrices
	Rplus.reserve(nU);  // Reserve space for nU copies of the 2x2 matrix

	Matrix2Xcd H = Matrix2Xcd::Zero(2, nU);

	// Declare the most time-critical part (memory access wise) (see https://www.cplusplus.com/forum/articles/7459/)
	// I don't know if this is optimal or not
	Matrix2cd** Ru = {};  // Will be an array of pointers
	Matrix2cd** Tu = {};
	Matrix2cd** Rd = {};
	Matrix2cd** Td = {};

	if (nLayers > 1) {
		// Allocate memory
		Ru = new Matrix2cd * [nU];
		Tu = new Matrix2cd * [nU];
		Rd = new Matrix2cd * [nU];
		Td = new Matrix2cd * [nU];
		for (int ii = 0; ii < nU; ii++)
		{
			Ru[ii] = new Matrix2cd[nLayers - 1];
			Tu[ii] = new Matrix2cd[nLayers - 1];
			Rd[ii] = new Matrix2cd[nLayers - 1];
			Td[ii] = new Matrix2cd[nLayers - 1];
		}
	}

	// Compute vertical slownesses of layers 1 to n for every u
	for (int kk = 0; kk < nLayers; kk++)
	{
		a.col(kk) = sqrt(pow(uP(kk), 2) - pow(u, 2));
		b.col(kk) = sqrt(pow(uS(kk), 2) - pow(u, 2));
	}
	
	complex<double> Hcoeff;
	Vector2cd Hvect;
	for (int ii = 0; ii < nU; ii++)
	{
		// Free surface
		Rplus[ii] = Rplus_freesurface(a(ii, 0), b(ii, 0), rhos(0), vS(0), u(ii));

		Hcoeff = 1. / (pow(1. - 2. * pow(vS(0), 2) * pow(u(ii), 2), 2) + 4. * pow(vS(0), 4) * pow(u(ii), 2) * a(ii, 0) * b(ii, 0));
		Hvect = Vector2cd((1. - 2. * pow(vS(0), 2) * pow(u(ii), 2)) * a(ii, 0),
			-2. * pow(vS(0), 2) * u(ii) * a(ii, 0) * b(ii, 0));
		H.col(ii) = Hcoeff * Hvect;

		// Precompute stuff for Rminus
		// start with ii = n - 1
		for (int kk = nLayers - 1 - 1; kk > -1; kk--)
		{
			computeRT(a(ii, kk), b(ii, kk), rhos(kk), vS(kk), a(ii, kk + 1), b(ii, kk + 1), rhos(kk + 1), vS(kk + 1), u(ii),
				Ru[ii][kk], Tu[ii][kk], Rd[ii][kk], Td[ii][kk]);
		}
	}

	// Pointer to the inverse of the basis matrix
	VectorXd*** Levin_basis = reinterpret_cast<VectorXd***>(Levin_basis_address);
	
	// Start the main loop ---------------------------------------------

	double practically_zero = 1e-30;
	int start_iter = 0;
	if (freq(0) <= practically_zero) {
		start_iter = 1;
		output.row(0) = VectorXcd::Zero(nRec);
	}
	
#pragma omp parallel for
	for (int jj = start_iter; jj < nF; jj++)
	{
		VectorXcd fpart = VectorXd::Zero(2 * nU);
		Matrix2cd Rminus = Matrix2cd::Zero(2, 2);
		Vector2cd V1;

		Vector2cd S_1u, S_1d;
		double omega = 2. * M_PI * freq(jj);
		
		for (int ii = 0; ii < nU; ii++)
		{
			// source terms
			complex<double> a_m = a(ii, 0);
			complex<double> b_m = b(ii, 0);
			complex<double> e_alpha = exp(1i * omega * a_m * (z_s - 0));
			complex<double> e_beta = exp(1i * omega * b_m * (z_s - 0));

			S_1u << -u(ii) / e_alpha,
				u(ii) * u(ii) / (b_m * e_beta);
			S_1d << u(ii) * e_alpha,
				u(ii) * u(ii) / b_m * e_beta;

			// reflectivity
			VectorXcd row_of_a = a.row(ii);
			VectorXcd row_of_b = b.row(ii);

			if (nLayers > 1) {
				computeRminus(Ru[ii], Tu[ii], Rd[ii], Td[ii], nLayers, row_of_a, row_of_b, d, omega, Rminus);
			}

			if (!direct) {
				if (!reverb) {
					V1 = Rminus * S_1d;  // no direct nor reverberations
				}
				else {
					V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (Rminus * S_1d); // no direct wave
				}
			}
			else if (!reverb) {
				V1 = S_1u + Rminus * S_1d;  // no reverberations
			}
			else {
				V1 = (Matrix2d::Identity() - Rminus * Rplus[ii]).inverse() * (S_1u + Rminus * S_1d);
			}
			fpart(ii) = 1i * 2. * (H(0, ii) * V1(0) + H(1, ii) * V1(1));
		}

		// Taper by the slowness window
		fpart.head(nU) = fpart.head(nU).array() * slowness_window;

		VectorXcd rhs = VectorXcd::Zero(2 * n);
		for (int rec = 0; rec < nRec; rec++)
		{
			// NOTE: could compute one bessel here to make the first interval better approximated
			//complex<double> integral = 0.5 * u(0) * bessel_J0(0) * fpart(0);
			complex<double> integral = 0;

			for (int q = 0; q < Q; q++)
			{
				rhs.head(n) = fpart.segment(q * n - q, n);

				integral += Levin_basis[jj][rec][q].dot(rhs);
			}

			// Scaling
			output(jj, rec) = omega * sourcefunction(jj) * integral / (4. * M_PI * rho_m);
		}
	}

	// Change to velocity or acceleration
	if (velocity) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * 1i * 2. * M_PI * freq.array();
		}
	}
	else if (acceleration) {
		for (int rec = 0; rec < nRec; rec++) {
			output.col(rec) = output.col(rec).array() * -4. * M_PI * M_PI * freq.array() * freq.array();
		}
	}
	
	if (nLayers > 1) {
		// Deallocate dynamic memory
		for (int ii = 0; ii < nU; ii++)
		{
			delete[] Ru[ii];
			delete[] Tu[ii];
			delete[] Rd[ii];
			delete[] Td[ii];
		}
		delete[] Ru;
		delete[] Tu;
		delete[] Rd;
		delete[] Td;
	}
	
	// Frequency windowing
	if (f_window) {
		ArrayXd freq_window = tukey(nF, 0.7);
		for (int rec = 0; rec < nRec; rec++)
		{
			output.col(rec) = output.col(rec).array() * freq_window;
		}
	}
}