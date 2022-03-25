#include <Eigen/Dense>

using namespace Eigen;

void compute_displ(Map<MatrixXd>& layers, Map<VectorXd>& freq,
	Map<VectorXcd>& source, Map<VectorXd>& receivers, Map<Vector4i>& options,
	Map<MatrixXcd>& output);

void compute_displ_Levin(Map<MatrixXd>& layers, Map<VectorXd>& freq,
	Map<VectorXcd>& source, Map<VectorXd>& receivers, Map<Vector4i>& options,
	Map<MatrixXcd>& output);

void basisfunctions(ArrayXd u, MatrixXd& basis, MatrixXd& basisp);

void basisfunctions_radial(ArrayXd u, MatrixXd& basis_out, MatrixXd& basisp_out);

void Q_slowness(VectorXd& vP, VectorXd& QP, VectorXd& vS, VectorXd& QS, double w,
	VectorXcd& uP, VectorXcd& uS, VectorXcd& vSQ);

Matrix2cd Rplus_freesurface(std::complex<double> a2, std::complex<double> b2, double rho2, std::complex<double> vS2, double u);

void computeRT(std::complex<double> a1, std::complex<double> b1, double rho1, std::complex<double> vS1,
	std::complex<double> a2, std::complex<double> b2, double rho2, std::complex<double> vS2, double u,
	Matrix2cd& Ru, Matrix2cd& Tu, Matrix2cd& Rd, Matrix2cd& Td);

void computeRminus(Matrix2cd* Ru, Matrix2cd* Tu, Matrix2cd* Rd, Matrix2cd* Td, int nLayers,
	VectorXcd& a, VectorXcd& b, VectorXd& d, double omega, Matrix2cd& Rminus);

ArrayXd tukey(int length, double alpha);

void compute_displ_Levin_precomp(uintptr_t Levin_basis_address, Map<ArrayXd>& u, Map<ArrayXd>& slowness_window, int n,
	Map<MatrixXd>& layers, Map<VectorXd>& freq, Map<VectorXcd>& source, Map<VectorXd>& receivers,
	Map<Vector4i>& options,	Map<MatrixXcd>& output);

uintptr_t precompute_Levin_basis(Map<VectorXd>& freq, Map<VectorXd>& receivers, Map<VectorXd>& u, int n_colloc);

void deallocate_Levin_basis(uintptr_t Levin_basis_address, int nF, int nRec);