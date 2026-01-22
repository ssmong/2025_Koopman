#ifndef MOVING_PULSATING_BALL_H
#define MOVING_PULSATING_BALL_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateEffector.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynParamManager.h"
#include <Eigen/Dense>
#include <string>

// [Validation] Payload for logging hidden states
#include "msgPayloadDefC/MPBMStateMsgPayload.h"
#include "architecture/messaging/messaging.h"

class MovingPulsatingBall : public StateEffector, public SysModel {
public:
    MovingPulsatingBall();
    ~MovingPulsatingBall();

    void registerStates(DynParamManager& states) override;
    void linkInStates(DynParamManager& states) override;
    void updateEffectorMassProps(double integTime) override;
    void computeDerivatives(double integTime, Eigen::Vector3d rDDot_BN_N, Eigen::Vector3d omegaDot_BN_B, Eigen::Vector3d sigma_BN) override;
    void updateEnergyMomContributions(double integTime, Eigen::Vector3d & rotAngMomPntCContr_B, double & rotEnergyContr, Eigen::Vector3d omega_BN_B) override;
    
    void Reset(uint64_t CurrentSimNanos) override;
    void UpdateState(uint64_t CurrentSimNanos) override;

public:
    // [Validation] Output Message for logging (r_slug, v_slug, T_interaction)
    Message<MPBMStateMsgPayload> mpbmOutMsg;

    // Parameters for 500kg Sat / 100kg Fuel
    double massInit;            //!< [kg] m_s: Fuel mass
    double radiusTank;          //!< [m] R: Tank radius
    double radiusSlugMin;       //!< [m] L_min: Frozen limit
    double kinematicViscosity;  //!< [m^2/s] Viscosity (Hydrazine)
    double rhoFluid;            //!< [kg/m^3] Fluid Density
    double surfaceTension;      //!< [N/m] Surface tension
    double t_sr;                //!< [-] Circulation factor
    
    // Penalty Parameters (Soft Barrier)
    double k_barrier;           //!< Stiffness for boundary enforcement
    double c_barrier;           //!< Damping for boundary enforcement

    Eigen::Vector3d r_TB_B;     //!< Tank Center w.r.t. Body Frame

    // [Init] Random Initialization Support
    Eigen::Vector3d r_Init_B;   //!< Initial Position
    Eigen::Vector3d v_Init_B;   //!< Initial Velocity
    Eigen::Vector3d omega_Init_B; //!< Initial Angular Velocity of the Ball

    std::string nameOfPosState;   
    std::string nameOfVelState;   
    std::string nameOfOmegaState; 

private:
    StateData *posState;    
    StateData *velState;    
    StateData *omegaState;  
    
    Eigen::Vector3d r_SB_B;     
    Eigen::Vector3d v_SB_B;     
    Eigen::Vector3d omega_BN_B; 
    
    double currentSlugRadius;   

    Eigen::Vector3d current_T_Li;
};

#endif