// by Qinghua Yu (org: nubot.trustie.net)

#ifndef ProbabilisticLoopClosing_HPP
#define ProbabilisticLoopClosing_HPP

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
namespace Eigen
{
    typedef Matrix<float, 6, 1> Vector6f;
    typedef Matrix<float, 1, 6> Vector6fRow;
    typedef Matrix<float, 6, 6> Matrix6f;
    typedef Matrix<double, 6, 1> Vector6d;
    typedef Matrix<double, 1, 6> Vector6dRow;
    typedef Matrix<double, 6, 6> Matrix6d;
}
namespace PLoopClosing {

///calculat the left Jacobian matrix of so3
//\mathbf J_l(\bm \phi) = \frac{\sin(\phi)}{\phi} \mathbf I_{3\times 3} + \frac{1-\cos(\phi)}{\phi}{\mathbf a^\wedge} +(1- \frac{\sin(\phi)}{\phi})\mathbf a \mathbf a^T
//\mathbf J_l^{-1}(\bm \phi) = \frac{\phi}{2}\cot(\frac{\phi}{2})\mathbf 1 - \frac{\phi}{2}{\mathbf a^\wedge} +(1-\frac{\phi}{2}\cot(\frac{\phi}{2}))\mathbf a \mathbf a^T
Eigen::Matrix3d leftJacobian(const Sophus::SO3d &_so3 );
Eigen::Matrix3d leftJacobian(const Eigen::Vector3d &_so3_vec );

///calculat the differential of $Jl(_so3)*_vec3$ to $_so3$
Eigen::Matrix3d differentialLeftJacobianCrossVec3(const Sophus::SO3d &_so3, const Eigen::Vector3d &_vec3 );
///calculat the differential of $Jl(_so3)^-1*_vec3$ to $_so3$
Eigen::Matrix3d differentialLeftJacobianInverseCrossVec3(const Sophus::SO3d &_so3, const Eigen::Vector3d &_vec3 );

///calculate the coviarance matrix of the pose with incremental model
Eigen::Matrix<double, 6, 6> incrementCoviarance(const Eigen::Matrix4d &Tf_pre, const Eigen::Matrix4d &Tf_inc, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &cov_pre, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &cov_inc);

///calculat the probability of the equation of two se3,
//using the Gaussian distribution model.
/* Let $\boldsymbol A=\boldsymbol\Sigma_{ij}^{-1}+\boldsymbol\Sigma_{co}^{-1}$，$\boldsymbol b=\left( \boldsymbol\Sigma_{ij}^{-1}+\boldsymbol\Sigma_{co}^{-1} \right)^{-1} \left( \boldsymbol\Sigma_{ij}^{-1}\hat{\boldsymbol\xi_{ij}}+\boldsymbol\Sigma_{co}^{-1}\hat{\boldsymbol\xi_j} \right)$，
 \begin{equation}
 \resizebox{.999\hsize}{!}{$
    \begin{aligned}
 & \int_{-\infty}^{\infty}  p(\boldsymbol\xi | \hat{\boldsymbol\xi_{ij}},\boldsymbol\Sigma_{ij}) p(\boldsymbol\xi | \hat{\boldsymbol\xi_j},\boldsymbol\Sigma_{co}) d\boldsymbol\xi  \cr
=& {1 \over 64\pi^6\boldsymbol|\Sigma_{ij}|^3\boldsymbol|\Sigma_{co}|^3 } \int_{-\infty}^{\infty} \!\! \exp\! \left\{\!\! -{1\over 2} \!
\left(  \boldsymbol\xi\! -\!\boldsymbol b \right)^T \!\!\boldsymbol A\! \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)
\!+\! {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}}
\!\right\} d\boldsymbol\xi \cr
=& { exp\left\{ {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}} \right\}
\over
64\pi^6\boldsymbol|\Sigma_{ij}|^3\boldsymbol|\Sigma_{co}|^3 }
\int_{-\infty}^{\infty} \!\! \exp\! \left\{\!\! -{1\over 2} \!
\left(  \boldsymbol\xi\! -\!\boldsymbol b \right)^T \!\!\boldsymbol A\! \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)
\!\right\} d\boldsymbol\xi \cr
=& { exp\left\{ {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}} \right\}
\over 64\pi^6\boldsymbol|\Sigma_{ij}|^3\boldsymbol|\Sigma_{co}|^3 }
(2\pi|\boldsymbol A|)^3
 \cr
\end{aligned}
$}
\end{equation}
*/
double integratTwoGaussianDistributions(const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose1, const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose2, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &_cov1, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &_cov2 );

double computLoopProbability(const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose_current, const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose_candidate, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_current, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_candidate, Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_covis);
bool outputEigenVaules(Eigen::Matrix6d _mat);
}
#endif
