%module movingPulsatingBall
%{
#include "movingPulsatingBall.h"
#include "msgPayloadDefC/MPBMStateMsgPayload.h"
%}

%include "swig_conly_data.i"
%include "swig_eigen.i"      // [추가] Eigen::Vector3d <-> Python List/Numpy 자동 변환 필수
%include "std_string.i"
%include "std_vector.i"

// 부모 클래스 포함 (상속 관계 인식용)
%include "sys_model.h"
%include "simulation/dynamics/_GeneralModuleFiles/stateEffector.h"

// 메시지 페이로드 정의
%include "msgPayloadDefC/MPBMStateMsgPayload.h"
struct MPBMOutMsg;

// 해당 모듈 헤더
%include "movingPulsatingBall.h"

%pythoncode %{
import sys
from Basilisk.architecture.swig_common_model import *
%}
