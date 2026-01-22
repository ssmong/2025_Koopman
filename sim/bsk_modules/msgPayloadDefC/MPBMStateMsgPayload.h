#ifndef MPBM_STATE_MSG_H
#define MPBM_STATE_MSG_H

/*! @brief Structure used to define the internal state of the Moving Pulsating Ball Model (MPBM). */
typedef struct {
    double r_Slug_B[3];      //!< [m] Slug position vector in Body frame components
    double v_Slug_B[3];      //!< [m/s] Slug velocity vector in Body frame components
    double T_Interaction[3]; //!< [N*m] Interaction torque exerted by the slug on the rigid body
    double mass;             //!< [kg] Mass of the fuel slug
}MPBMStateMsgPayload;

#endif