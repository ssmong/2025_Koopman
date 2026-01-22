#include "movingPulsatingBall.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/avsEigenSupport.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

Eigen::Vector3d softSaturate(Eigen::Vector3d vec, double limit) {
    double norm = vec.norm();
    if (norm < 1e-9) return vec;
    double saturated_norm = limit * std::tanh(norm / limit);
    return vec * (saturated_norm / norm);
}

Eigen::Matrix3d eigenTilde(Eigen::Vector3d v) {
    Eigen::Matrix3d m;
    m << 0, -v[2], v[1],
       v[2], 0, -v[0],
       -v[1], v[0], 0;
    return m;
}

MovingPulsatingBall::MovingPulsatingBall() {
    this->massInit = 100.0;
    this->radiusTank = 0.50;
    this->radiusSlugMin = 0.10;
    this->kinematicViscosity = 2.839e-6; 
    this->rhoFluid = 1004.0; 
    this->surfaceTension = 0.066;
    
    this->t_sr = 0.1; 
    
    this->k_barrier = 0.0; 
    this->c_barrier = 0.0;

    this->r_TB_B.setZero();
    this->r_Init_B << 0.0, 0.0, 0.1;
    this->v_Init_B.setZero();
    this->omega_Init_B.setZero();
    
    this->nameOfPosState = "mpbmPos";
    this->nameOfVelState = "mpbmVel";
    this->nameOfOmegaState = "mpbmOmega";
    
    this->current_T_Li.setZero();
}

MovingPulsatingBall::~MovingPulsatingBall() {}

void MovingPulsatingBall::Reset(uint64_t CurrentSimNanos) {
    this->effProps.mEff = this->massInit;
    this->current_T_Li.setZero();
    
    MPBMStateMsgPayload outMsgBuffer = {}; 
    eigenVector3d2CArray(this->r_Init_B, outMsgBuffer.r_Slug_B);
    eigenVector3d2CArray(this->v_Init_B, outMsgBuffer.v_Slug_B);
    eigenVector3d2CArray(this->current_T_Li, outMsgBuffer.T_Interaction);
    outMsgBuffer.mass = this->massInit;
    this->mpbmOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}

void MovingPulsatingBall::registerStates(DynParamManager& states) {
    this->posState = states.registerState(3, 1, this->nameOfPosState);
    this->posState->setState(this->r_Init_B);
    this->velState = states.registerState(3, 1, this->nameOfVelState);
    this->velState->setState(this->v_Init_B);
    this->omegaState = states.registerState(3, 1, this->nameOfOmegaState);
    this->omegaState->setState(this->omega_Init_B);
}

void MovingPulsatingBall::linkInStates(DynParamManager& states) {}

void MovingPulsatingBall::updateEffectorMassProps(double integTime) {
    Eigen::Vector3d r_rel = this->posState->getState();
    Eigen::Vector3d v_rel = this->velState->getState();
    
    double r_norm = r_rel.norm();
    double maxCenterRadius = this->radiusTank - this->radiusSlugMin;
    
    // Wall correction: Impulse-based collision response with energy loss
    if (r_norm > maxCenterRadius) {
       Eigen::Vector3d n_hat = r_rel.normalized();
       
       // 1. Position Clamping to prevent tunneling
       r_norm = maxCenterRadius;
       r_rel = n_hat * r_norm;
       
       // 2. Velocity Reflection
       double v_radial = v_rel.dot(n_hat);
       if (v_radial > 0) { 
           // Coefficient of Restitution e = 0.5 (Energy dissipation)
           // v_new = v_old - (1 + e) * v_radial * n_hat
           v_rel -= 1.5 * v_radial * n_hat;
       }

       this->posState->setState(r_rel);
       this->velState->setState(v_rel);
    }
    
    this->currentSlugRadius = this->radiusTank - r_norm;
    this->r_SB_B = this->r_TB_B + r_rel;
    this->v_SB_B = v_rel;
    
    this->effProps.mEff = this->massInit;
    this->effProps.rEff_CB_B = this->r_SB_B;
    this->effProps.rEffPrime_CB_B = this->v_SB_B;
    
    double I_slug_scalar = 0.4 * this->massInit * std::pow(this->currentSlugRadius, 2);
    Eigen::Matrix3d I_slug_matrix = I_slug_scalar * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rTilde = eigenTilde(this->r_SB_B);
    this->effProps.IEffPntB_B = I_slug_matrix - this->massInit * rTilde * rTilde;
    
    double r_dot_scalar = (r_norm > 1e-6) ? r_rel.dot(v_rel) / r_norm : 0.0;
    double L_dot = -r_dot_scalar;
    
    double I_slug_dot_scalar = 0.8 * this->massInit * this->currentSlugRadius * L_dot;
    Eigen::Matrix3d I_slug_dot_matrix = I_slug_dot_scalar * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d vTilde = eigenTilde(this->v_SB_B);
    this->effProps.IEffPrimePntB_B = I_slug_dot_matrix - this->massInit * (vTilde * rTilde + rTilde * vTilde);
}

void MovingPulsatingBall::updateEnergyMomContributions(double integTime, Eigen::Vector3d & rotAngMomPntCContr_B, double & rotEnergyContr, Eigen::Vector3d omega_BN_B) {
    this->omega_BN_B = omega_BN_B;
    Eigen::Vector3d omega_s = this->omegaState->getState();
    double I_slug_scalar = 0.4 * this->massInit * std::pow(this->currentSlugRadius, 2);
    Eigen::Matrix3d I_slug = I_slug_scalar * Eigen::Matrix3d::Identity();
    Eigen::Vector3d H_internal = I_slug * omega_s + this->massInit * this->r_SB_B.cross(this->v_SB_B);
    rotAngMomPntCContr_B = this->effProps.IEffPntB_B * omega_BN_B + H_internal;
    rotEnergyContr = 0.0;
}

void MovingPulsatingBall::computeDerivatives(double integTime, Eigen::Vector3d rDDot_BN_N, Eigen::Vector3d omegaDot_BN_B, Eigen::Vector3d sigma_BN) {
    Eigen::Vector3d r_vec = this->posState->getState();
    Eigen::Vector3d v_vec = this->velState->getState();
    Eigen::Vector3d omega_s = this->omegaState->getState();
    Eigen::Vector3d omega_hub = this->omega_BN_B;
    Eigen::Vector3d omegaDot_hub = omegaDot_BN_B;
    
    double r_norm = r_vec.norm();
    if (r_norm < 1e-4) {
       r_vec = (r_norm < 1e-6) ? Eigen::Vector3d(0,0,1) * 1e-4 : r_vec.normalized() * 1e-4;
       r_norm = 1e-4;
    }
    
    double L = this->radiusTank - r_norm;
    Eigen::Vector3d e_i = r_vec.normalized();
    double r_dot_scalar = r_vec.dot(v_vec) / r_norm;
    double L_calc = (L < this->radiusSlugMin) ? this->radiusSlugMin : L;

    // Kinematics
    Eigen::Vector3d w_i = e_i.cross(v_vec) / r_norm;
    Eigen::Vector3d omega_ri = e_i.dot(omega_s) * e_i;

    // Raw Forces (Normal Force N)
    Eigen::Vector3d term1_vec = e_i.cross(omega_hub + w_i);
    double term1 = r_norm * term1_vec.squaredNorm();
    double term2 = (omega_hub.cross(this->r_TB_B)).dot(omega_hub.cross(e_i));
    double term3 = -(this->r_TB_B.cross(e_i)).dot(omegaDot_hub);
    Eigen::Vector3d total_omega = omega_s + omega_hub;
    double term4 = (this->massInit / 4.0) * L * total_omega.squaredNorm();
    double term5 = -5.0 * M_PI * this->surfaceTension * L;
    
    double N_val_raw = (3.0 * this->massInit / 8.0) * (term1 + term2 + term3) + term4 + term5;

    // Friction Force (F_b)
    double coef_F = (6500.0 * this->kinematicViscosity * this->massInit) / (L_calc * L_calc);
    Eigen::Vector3d inner_vec = e_i.cross(v_vec) + L_calc * omega_s;
    Eigen::Vector3d F_bi_raw = -coef_F * e_i.cross(inner_vec);
    
    // Interaction Torque (T_L)
    double omega_s_norm = omega_s.norm();
    if(omega_s_norm < 1e-8) omega_s_norm = 1e-8;

    double f_ac = 0.36 * std::pow(this->massInit, 4.0/3.0) 
                * std::pow(this->rhoFluid, 1.0/6.0) 
                * std::sqrt(this->kinematicViscosity) 
                * std::pow(L_calc / this->radiusSlugMin, 2.0) 
                * std::sqrt(omega_s_norm);
    
    Eigen::Vector3d T_Li_raw = f_ac * (this->t_sr * omega_ri + (1.0 - this->t_sr)*(omega_s - omega_ri));

    // Soft Saturation (Increased limits for physical realism)
    Eigen::Vector3d T_Li_Physical = softSaturate(T_Li_raw, 10.0);
    Eigen::Vector3d F_bi_Physical = softSaturate(F_bi_raw, 100.0);
    double N_val_Physical = 1000.0 * std::tanh(N_val_raw / 1000.0);

    Eigen::Vector3d F_Li_Physical = N_val_Physical * e_i + F_bi_Physical;

    // Solve Dynamics
    Eigen::MRPd sigmaMRP(sigma_BN);
    Eigen::Matrix3d dcm_BN = sigmaMRP.toRotationMatrix().transpose();
    Eigen::Vector3d a_hub_B = dcm_BN * rDDot_BN_N;
    Eigen::Vector3d r_Si = this->r_TB_B + r_vec; 
    Eigen::Vector3d inertial_acc = a_hub_B + omegaDot_hub.cross(r_Si) + omega_hub.cross(omega_hub.cross(r_Si)) + 2.0 * omega_hub.cross(v_vec);
    
    Eigen::Vector3d v_dot = (-F_Li_Physical / this->massInit) - inertial_acc;
    
    double I_s = 0.4 * this->massInit * L_calc * L_calc;
    if (I_s < 1e-6) I_s = 1e-6;

    Eigen::Vector3d RHS_Eq4 = -T_Li_Physical - L_calc * e_i.cross(F_Li_Physical);
    Eigen::Vector3d LHS_Coriolis = (0.4 * this->massInit * L_calc) * (-2.0 * r_dot_scalar) * (omega_s + omega_hub);
    Eigen::Vector3d LHS_Hub = I_s * (omegaDot_hub + omega_hub.cross(omega_s));
    
    Eigen::Vector3d Torque_Net = RHS_Eq4 - LHS_Coriolis - LHS_Hub;
    Eigen::Vector3d omega_s_dot = Torque_Net / I_s;

    double omega_s_dot_norm = omega_s_dot.norm();
    if (omega_s_dot_norm > 100.0) {
       omega_s_dot = omega_s_dot * (100.0 / omega_s_dot_norm);
    }

    this->posState->setDerivative(v_vec);
    this->velState->setDerivative(v_dot);
    this->omegaState->setDerivative(omega_s_dot);
    
    this->current_T_Li = T_Li_Physical;
}

void MovingPulsatingBall::UpdateState(uint64_t CurrentSimNanos) {
    MPBMStateMsgPayload outMsgBuffer = {}; 
    Eigen::Vector3d r_vec = this->posState->getState();
    Eigen::Vector3d v_vec = this->velState->getState();
    Eigen::Vector3d t_vec = this->current_T_Li;

    eigenVector3d2CArray(r_vec, outMsgBuffer.r_Slug_B);
    eigenVector3d2CArray(v_vec, outMsgBuffer.v_Slug_B);
    eigenVector3d2CArray(t_vec, outMsgBuffer.T_Interaction);
    outMsgBuffer.mass = this->massInit;
    
    this->mpbmOutMsg.write(&outMsgBuffer, this->moduleID, CurrentSimNanos);
}