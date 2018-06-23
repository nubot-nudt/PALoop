// by Qinghua Yu (org: nubot.trustie.net)

#include "ProbabilisticLoopClosing.h"
#include <iostream>
//namespace Eigen {
//void symmetrizeMatrix(Eigen::Matrix<double, 6, 6> &_mat)//强制转换为对称矩阵
//{
//    _mat(1,0) = _mat(0,1);
//    _mat(2,0) = _mat(0,2); _mat(2,1) = _mat(1,2);
//    _mat(3,0) = _mat(0,3); _mat(3,1) = _mat(1,3); _mat(3,2) = _mat(2,3);
//    _mat(4,0) = _mat(0,4); _mat(4,1) = _mat(1,4); _mat(4,2) = _mat(2,4); _mat(4,3) = _mat(3,4);
//    _mat(5,0) = _mat(0,5); _mat(5,1) = _mat(1,5); _mat(5,2) = _mat(2,5); _mat(5,3) = _mat(3,5); _mat(5,4) = _mat(4,5);
//}
//}
namespace PLoopClosing {

///calculat the left Jacobian matrix of so3
//\mathbf J_l(\bm \phi) = \frac{\sin(\phi)}{\phi} \mathbf I_{3\times 3} + \frac{1-\cos(\phi)}{\phi}{\mathbf a^\wedge} +(1- \frac{\sin(\phi)}{\phi})\mathbf a \mathbf a^T
//\mathbf J_l^{-1}(\bm \phi) = \frac{\phi}{2}\cot(\frac{\phi}{2})\mathbf 1 - \frac{\phi}{2}{\mathbf a^\wedge} +(1-\frac{\phi}{2}\cot(\frac{\phi}{2}))\mathbf a \mathbf a^T
Eigen::Matrix3d leftJacobian(const Sophus::SO3d &_so3 )
{
    double theta;
    Eigen::Vector3d vec = Sophus::SO3d::logAndTheta(_so3,&theta);
    if(theta==0)
        return Eigen::Matrix3d::Identity();
    vec.normalize(); // inplace
    double sintheta_theta = std::sin(theta)/theta;
    Eigen::Matrix3d left_jac = Eigen::Matrix3d::Identity()*sintheta_theta
             + Sophus::SO3d::hat(vec)*((1-std::cos(theta))/theta)
             + (1-sintheta_theta)*vec*vec.transpose();
    return left_jac;
}
Eigen::Matrix3d leftJacobian(const Eigen::Vector3d &_so3_vec )
{
    double theta = _so3_vec.norm();
    if(theta==0 || theta==M_PI)
        return Eigen::Matrix3d::Identity();
    Eigen::Vector3d vec_norm = _so3_vec.normalized();
    double sintheta_theta = std::sin(theta)/theta;
    Eigen::Matrix3d left_jac = Eigen::Matrix3d::Identity()*sintheta_theta
             + Sophus::SO3d::hat(vec_norm)*((1-std::cos(theta))/theta)
             + (1-sintheta_theta)*vec_norm*vec_norm.transpose();
    return left_jac;
}
///calculat the differential of $Jl(_so3)*_vec3$ to $_so3$
Eigen::Matrix3d differentialLeftJacobianCrossVec3(const Sophus::SO3d &_so3, const Eigen::Vector3d &_vec3 )
{
    double theta;
    Eigen::Vector3d vec = Sophus::SO3d::logAndTheta(_so3,&theta);
    if(theta==0)
        return Eigen::Matrix3d::Zero();
    Eigen::Vector3d vec_norm = vec.normalized();
    double sintheta = std::sin(theta), costheta = std::cos(theta);
    double _1_theta2 = 1.0/(theta*theta);
    Eigen::Matrix3d PJlPTheta = (costheta*theta-sintheta)*_1_theta2*(Eigen::Matrix3d::Identity()+vec_norm*vec_norm.transpose())
                              + Sophus::SO3d::hat(vec_norm)*(sintheta*theta-1+costheta)*_1_theta2;
    return PJlPTheta*_vec3/theta*vec.transpose();
}
///calculat the differential of $Jl(_so3)^-1*_vec3$ to $_so3$
Eigen::Matrix3d differentialLeftJacobianInverseCrossVec3(const Sophus::SO3d &_so3, const Eigen::Vector3d &_vec3 )
{
    double theta;
    Eigen::Vector3d vec = Sophus::SO3d::logAndTheta(_so3,&theta);
    if(theta==0)
        return Eigen::Matrix3d::Zero();
    Eigen::Vector3d vec_norm = vec.normalized();
    double sintheta_2 = std::sin(theta/2);
    double cottheta_2 = 1/std::tan(theta/2);
    Eigen::Matrix3d PJlinvPTheta = (cottheta_2/2-theta/4/(sintheta_2*sintheta_2))*(Eigen::Matrix3d::Identity()-vec_norm*vec_norm.transpose())
                              - Sophus::SO3d::hat(vec_norm)/2;
    return PJlinvPTheta*_vec3/theta*vec.transpose();
}

///calculate the coviarance matrix of the pose with incremental model
Eigen::Matrix<double, 6, 6> incrementCoviarance(const Eigen::Matrix4d &Tf_pre, const Eigen::Matrix4d &Tf_inc, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &cov_pre, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &cov_inc)
{
    Sophus::SO3d so3_pre(Tf_pre.topLeftCorner<3,3>()), so3_inc(Tf_inc.topLeftCorner<3,3>());
    Eigen::Vector3d so3_aft = so3_pre.log()+leftJacobian( so3_pre )*so3_inc.log();//如果直接用矩阵计算变化后的Tf，可能导致前后跨越Pi，计算出现奇异。因此这里使用一阶近似增量计算方法
    Eigen::Matrix3d Jl_pre = leftJacobian( so3_pre );
    Eigen::Matrix3d Jl_aft = leftJacobian( so3_aft );
    Eigen::Matrix3d Jl_inc = leftJacobian( so3_inc );
    Eigen::Matrix3d Jl_aft_inv = Jl_aft.inverse();
    Eigen::Matrix<double, 6, 6> PaftPpre, PaftPinc;//{\partial \boldsymbol\xi_2 \over \partial\boldsymbol\xi_{1}^T} and {\partial \boldsymbol\xi_2 \over \partial\boldsymbol\xi_{21}^T}
    PaftPpre.topLeftCorner<3,3>() = Eigen::Matrix3d::Identity() + differentialLeftJacobianInverseCrossVec3( so3_pre, so3_inc.log() );
    PaftPpre.topRightCorner<3,3>().setZero();
    PaftPpre.bottomLeftCorner<3,3>().setZero();
    PaftPpre.bottomRightCorner<3,3>() = Jl_aft_inv*Tf_inc.topLeftCorner<3,3>()*Jl_pre;
    PaftPinc.topLeftCorner<3,3>() = Jl_pre.inverse();
    PaftPinc.topRightCorner<3,3>().setZero();
    PaftPinc.bottomLeftCorner<3,3>() = -Jl_aft_inv * Sophus::SO3::hat(Tf_inc.topLeftCorner<3,3>()*Jl_pre*Tf_pre.topRightCorner<3,1>())*Jl_inc//;
            + Jl_aft_inv*differentialLeftJacobianCrossVec3( so3_inc, Tf_inc.topRightCorner<3,1>() );
    PaftPinc.bottomRightCorner<3,3>() = Jl_aft_inv*Jl_inc;
    return PaftPpre*cov_pre*PaftPpre.transpose() + PaftPinc*cov_inc*PaftPinc.transpose();
}

///calculat the probability of the equation of two se3,
double integratTwoGaussianDistributions(const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose1, const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose2, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &_cov1, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> &_cov2 )
{//using the Gaussian distribution model.
    /* Let $\boldsymbol A=\boldsymbol\Sigma_{ij}^{-1}+\boldsymbol\Sigma_{co}^{-1}$，$\boldsymbol b=\left( \boldsymbol\Sigma_{ij}^{-1}+\boldsymbol\Sigma_{co}^{-1} \right)^{-1} \left( \boldsymbol\Sigma_{ij}^{-1}\hat{\boldsymbol\xi_{ij}}+\boldsymbol\Sigma_{co}^{-1}\hat{\boldsymbol\xi_j} \right)$，
     \begin{equation}
     \resizebox{.999\hsize}{!}{$
        \begin{aligned}
        & \int_{-\infty}^{\infty}  p(\boldsymbol\xi | \hat{\boldsymbol\xi_{ij}},\boldsymbol\Sigma_{ij}) p(\boldsymbol\xi | \hat{\boldsymbol\xi_j},\boldsymbol\Sigma_{co}) d\boldsymbol\xi  \cr
        =& {1 \over 64\pi^6\boldsymbol|\Sigma_{ij}|^{1/2}\boldsymbol|\Sigma_{co}|^{1/2} } \int_{-\infty}^{\infty} \!\! \exp\! \left\{\!\! -{1\over 2} \!
        \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)^T \!\!\boldsymbol A\! \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)
        \!+\! {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}}
        \!\right\} d\boldsymbol\xi \cr
        =& { exp\left\{ {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}} \right\}
            \over
            64\pi^6\boldsymbol|\Sigma_{ij}|^{1/2}\boldsymbol|\Sigma_{co}|^{1/2} }
        \int_{-\infty}^{\infty} \!\! \exp\! \left\{\!\! -{1\over 2} \!
        \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)^T \!\!\boldsymbol A\! \left(  \boldsymbol\xi\! -\!\boldsymbol b \right)
        \!\right\} d\boldsymbol\xi \cr
        =& { exp\left\{ {1\over 2}\boldsymbol b^T \!\boldsymbol A \boldsymbol b  \!-\! {1\over 2} \hat{\boldsymbol\xi_{ij}^T}\! \boldsymbol\Sigma_{ij}^{-1} \!  \hat{\boldsymbol\xi_{ij}} - {1\over 2} \hat{\boldsymbol\xi_{j}^T}\! \boldsymbol\Sigma_{co}^{-1} \!  \hat{\boldsymbol\xi_{j}} \right\}
            \over 64\pi^6\boldsymbol|\Sigma_{ij}|^{1/2}\boldsymbol|\Sigma_{co}|^{1/2} }
        8\pi^3|\boldsymbol A^{-1}|^{1/2}
        \cr
        \end{aligned}
        $}
    */
    double p;
    Eigen::Matrix<double, 6, 6> cov1_inv = _cov1.inverse();
    Eigen::Matrix<double, 6, 6> cov2_inv = _cov2.inverse();
    Eigen::Matrix<double, 6, 6> A = cov1_inv + cov2_inv;
    Eigen::Matrix<double, 6, 6> A_inv = A.inverse();
    const Eigen::Matrix<double, 6, 1> b = A_inv * ( cov1_inv*_pose1 + cov2_inv*_pose2 );
    Eigen::Matrix<double, 1, 1> temp = b.transpose()*A*b - _pose1.transpose()*cov1_inv*_pose1 - _pose2.transpose()*cov2_inv*_pose2;
    if(temp(0)>0)//this should not happen
    {
        std::cout <<"\033[31mtemp=" << temp << "\033[m" << std::endl;
        temp(0) = 0;
    }
    p = std::exp(0.5*temp(0)) * std::sqrt( A_inv.norm()/_cov1.norm()/_cov2.norm() ) / (8*M_PI*M_PI*M_PI);
    return p;
}

double computLoopProbability(const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose_current, const Eigen::Matrix<double, 6, 1,Eigen::DontAlign> &_pose_candidate, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_current, const Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_candidate, Eigen::Matrix<double, 6, 6,Eigen::DontAlign> _cov_covis)
{/*
    Sophus::SE3d pose_current   = Sophus::SE3d::exp(_pose_current);
    Sophus::SE3d pose_candidate = Sophus::SE3d::exp(_pose_candidate);
    Sophus::SE3d pose_diff      = pose_current*pose_candidate.inverse();
    Eigen::Vector3d t_candidate = pose_candidate.translation();
    Eigen::Matrix3d R_diff      = pose_diff.so3().matrix();
    Eigen::Matrix3d Jl_pre = leftJacobian( pose_candidate.so3() );
    Eigen::Matrix3d Jl_aft = leftJacobian( pose_current.so3() );
    Eigen::Matrix3d Jl_inc = leftJacobian( pose_diff.so3() );
    Eigen::Matrix3d Jl_aft_inv = Jl_aft.inverse();
    Eigen::Matrix<double, 6, 6> PaftPpre, PaftPinc;//{\partial \boldsymbol\xi_2 \over \partial\boldsymbol\xi_{1}^T} and {\partial \boldsymbol\xi_2 \over \partial\boldsymbol\xi_{21}^T}
    PaftPpre.topLeftCorner<3,3>().setIdentity();
    PaftPpre.topRightCorner<3,3>().setZero();
    PaftPpre.bottomLeftCorner<3,3>().setZero();
    PaftPpre.bottomRightCorner<3,3>() = Jl_aft_inv*R_diff*Jl_pre;
    PaftPinc.topLeftCorner<3,3>() = Jl_pre.inverse();
    PaftPinc.topRightCorner<3,3>().setZero();
    PaftPinc.bottomLeftCorner<3,3>() = -Jl_aft_inv * Sophus::SO3::hat(R_diff*Jl_pre*t_candidate)*Jl_inc;
    PaftPinc.bottomRightCorner<3,3>() = Jl_aft_inv*Jl_inc;
    Eigen::Matrix<double, 6, 6> PaftPinc_inv = PaftPinc.inverse();
    std::cout <<"sigma2_diff"<<std::flush;
    Eigen::Matrix6d a = PaftPpre*_cov_candidate*PaftPpre.transpose();
    std::cout <<"1"<<std::flush;
    Eigen::Matrix6d b = _cov_current - a;///sigma2_diff 怎么算都不对，如果用减号，极易出现非正定。
    std::cout <<"2"<<std::flush;
    Eigen::Matrix6d sigma2_diff = PaftPinc_inv*b*PaftPinc_inv.transpose();
//    Eigen::Matrix6d sigma2_diff = _cov_current;
    std::cout <<"特征值="<<std::flush, PLoopClosing::outputEigenVaules(sigma2_diff);
    double p = PLoopClosing::integratTwoGaussianDistributions( pose_diff.log(), Eigen::Vector6d::Zero(), sigma2_diff, _cov_covis );
*/
    if( _cov_covis==Eigen::Matrix6d::Zero() )
        return 0;
    if( _cov_current==Eigen::Matrix6d::Zero() )
    {
        Eigen::Matrix<double, 1, 1> temp = (_pose_current-_pose_candidate).transpose()*_cov_covis*(_pose_current-_pose_candidate);
        return std::exp(-0.5*temp(0));
    }
    double p = PLoopClosing::integratTwoGaussianDistributions( _pose_candidate, _pose_current, _cov_covis, _cov_current );
    p *= (8*M_PI*M_PI*M_PI) * std::sqrt(_cov_covis.norm());//parameter of c_1 in the thesis
    return p;
}
//std::ostream& operator<< outputEigenVaules(Eigen::Matrix6d _mat)
//{
//    Eigen::SelfAdjointEigenSolver<Eigen::Matrix6d> eigenSolver(_mat);
//    if (eigenSolver.info() == Eigen::Success)
//    {
//        bool is_positive = true;
//        for(int i=0;i<6;i++)
//            if(eigenSolver.eigenvalues()(i)<0)
//            {is_positive = false; break;}
//        if(is_positive)
//            std::cout << eigenSolver.eigenvalues().transpose()  << std::endl;
//        else
//            std::cout <<"\033[31m" << eigenSolver.eigenvalues().transpose() << "\033[m" << std::endl;
//        return true;
//    }
//    else
//        return false;
//}
}
