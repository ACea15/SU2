/*!
 * \file CSolver.cpp
 * \brief Main subroutines for CSolver class.
 * \author F. Palacios, T. Economon
 * \version 7.3.1 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2022, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */


#include "../../include/solvers/CSolver.hpp"
#include "../../include/gradients/computeGradientsGreenGauss.hpp"
#include "../../include/gradients/computeGradientsLeastSquares.hpp"
#include "../../include/limiters/computeLimiters.hpp"
#include "../../../Common/include/toolboxes/MMS/CIncTGVSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CInviscidVortexSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSIncEulerSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSIncNSSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSNSTwoHalfCirclesSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSNSTwoHalfSpheresSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSNSUnitQuadSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CMMSNSUnitQuadSolutionWallBC.hpp"
#include "../../../Common/include/toolboxes/MMS/CNSUnitQuadSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CRinglebSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CTGVSolution.hpp"
#include "../../../Common/include/toolboxes/MMS/CUserDefinedSolution.hpp"
#include "../../../Common/include/toolboxes/printing_toolbox.hpp"
#include "../../../Common/include/toolboxes/C1DInterpolation.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"
#include "../../../Common/include/toolboxes/CLinearPartitioner.hpp"
#include "../../../Common/include/adt/CADTPointsOnlyClass.hpp"
#include "../../include/CMarkerProfileReaderFVM.hpp"


CSolver::CSolver(LINEAR_SOLVER_MODE linear_solver_mode) : System(linear_solver_mode) {

  rank = SU2_MPI::GetRank();
  size = SU2_MPI::GetSize();

  adjoint = false;

  /*--- Set the multigrid level to the finest grid. This can be
        overwritten in the constructors of the derived classes. ---*/
  MGLevel = MESH_0;

  /*--- Array initialization ---*/

  OutputHeadingNames = nullptr;
  Residual           = nullptr;
  Residual_i         = nullptr;
  Residual_j         = nullptr;
  Solution           = nullptr;
  Solution_i         = nullptr;
  Solution_j         = nullptr;
  Vector             = nullptr;
  Vector_i           = nullptr;
  Vector_j           = nullptr;
  Res_Conv           = nullptr;
  Res_Visc           = nullptr;
  Res_Sour           = nullptr;
  Res_Conv_i         = nullptr;
  Res_Visc_i         = nullptr;
  Res_Conv_j         = nullptr;
  Res_Visc_j         = nullptr;
  Jacobian_i         = nullptr;
  Jacobian_j         = nullptr;
  Jacobian_ii        = nullptr;
  Jacobian_ij        = nullptr;
  Jacobian_ji        = nullptr;
  Jacobian_jj        = nullptr;
  Restart_Vars       = nullptr;
  Restart_Data       = nullptr;
  base_nodes         = nullptr;
  nOutputVariables   = 0;
  ResLinSolver       = 0.0;

  /*--- Variable initialization to avoid valgrid warnings when not used. ---*/

  IterLinSolver = 0;

  /*--- Initialize pointer for any verification solution. ---*/
  VerificationSolution  = nullptr;

  /*--- Flags for the periodic BC communications. ---*/

  rotate_periodic   = false;
  implicit_periodic = false;

  /*--- Containers to store the markers. ---*/
  nMarker = 0;

  /*--- Flags for the dynamic grid (rigid movement or unsteady deformation). ---*/
  dynamic_grid = false;

  /*--- Auxiliary data needed for CFL adaption. ---*/

  Old_Func = 0;
  New_Func = 0;
  NonLinRes_Counter = 0;

  nPrimVarGrad = 0;
  nPrimVar     = 0;

}

CSolver::~CSolver(void) {

  unsigned short iVar;

  /*--- Public variables, may be accessible outside ---*/

  delete [] OutputHeadingNames;

  /*--- Private ---*/

  delete [] Residual;
  delete [] Residual_i;
  delete [] Residual_j;
  delete [] Solution;
  delete [] Solution_i;
  delete [] Solution_j;
  delete [] Vector;
  delete [] Vector_i;
  delete [] Vector_j;
  delete [] Res_Conv;
  delete [] Res_Visc;
  delete [] Res_Sour;
  delete [] Res_Conv_i;
  delete [] Res_Visc_i;
  delete [] Res_Visc_j;

  if (Jacobian_i != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_i[iVar];
    delete [] Jacobian_i;
  }

  if (Jacobian_j != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_j[iVar];
    delete [] Jacobian_j;
  }

  if (Jacobian_ii != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_ii[iVar];
    delete [] Jacobian_ii;
  }

  if (Jacobian_ij != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_ij[iVar];
    delete [] Jacobian_ij;
  }

  if (Jacobian_ji != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_ji[iVar];
    delete [] Jacobian_ji;
  }

  if (Jacobian_jj != nullptr) {
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_jj[iVar];
    delete [] Jacobian_jj;
  }

  delete [] Restart_Vars;
  delete [] Restart_Data;

  delete VerificationSolution;
}

void CSolver::GetPeriodicCommCountAndType(const CConfig* config,
                                          unsigned short commType,
                                          unsigned short &COUNT_PER_POINT,
                                          unsigned short &MPI_TYPE,
                                          unsigned short &ICOUNT,
                                          unsigned short &JCOUNT) const {
  switch (commType) {
    case PERIODIC_VOLUME:
      COUNT_PER_POINT  = 1;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_NEIGHBORS:
      COUNT_PER_POINT  = 1;
      MPI_TYPE         = COMM_TYPE_UNSIGNED_SHORT;
      break;
    case PERIODIC_RESIDUAL:
      COUNT_PER_POINT  = nVar + nVar*nVar + 1;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_IMPLICIT:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_LAPLACIAN:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_MAX_EIG:
      COUNT_PER_POINT  = 1;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_SENSOR:
      COUNT_PER_POINT  = 2;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PERIODIC_SOL_GG:
    case PERIODIC_SOL_GG_R:
      COUNT_PER_POINT  = nVar*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nVar;
      JCOUNT           = nDim;
      break;
    case PERIODIC_PRIM_GG:
    case PERIODIC_PRIM_GG_R:
      COUNT_PER_POINT  = nPrimVarGrad*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nPrimVarGrad;
      JCOUNT           = nDim;
      break;
    case PERIODIC_SOL_LS:
    case PERIODIC_SOL_ULS:
    case PERIODIC_SOL_LS_R:
    case PERIODIC_SOL_ULS_R:
      COUNT_PER_POINT  = nDim*nDim + nVar*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nVar;
      JCOUNT           = nDim;
      break;
    case PERIODIC_PRIM_LS:
    case PERIODIC_PRIM_ULS:
    case PERIODIC_PRIM_LS_R:
    case PERIODIC_PRIM_ULS_R:
      COUNT_PER_POINT  = nDim*nDim + nPrimVarGrad*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nPrimVarGrad;
      JCOUNT           = nDim;
      break;
    case PERIODIC_LIM_PRIM_1:
      COUNT_PER_POINT  = nPrimVarGrad*2;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nPrimVarGrad;
      break;
    case PERIODIC_LIM_PRIM_2:
      COUNT_PER_POINT  = nPrimVarGrad;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nPrimVarGrad;
      break;
    case PERIODIC_LIM_SOL_1:
      COUNT_PER_POINT  = nVar*2;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nVar;
      break;
    case PERIODIC_LIM_SOL_2:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      ICOUNT           = nVar;
      break;
    default:
      SU2_MPI::Error("Unrecognized quantity for periodic communication.",
                     CURRENT_FUNCTION);
      break;
  }
}

namespace PeriodicCommHelpers {
  CVectorOfMatrix& selectGradient(CVariable* nodes, unsigned short commType) {
    switch(commType) {
      case PERIODIC_PRIM_GG:
      case PERIODIC_PRIM_LS:
      case PERIODIC_PRIM_ULS:
        return nodes->GetGradient_Primitive();
        break;
      case PERIODIC_SOL_GG:
      case PERIODIC_SOL_LS:
      case PERIODIC_SOL_ULS:
        return nodes->GetGradient();
        break;
      default:
        return nodes->GetGradient_Reconstruction();
        break;
    }
  }

  const su2activematrix& selectField(CVariable* nodes, unsigned short commType) {
    switch(commType) {
      case PERIODIC_PRIM_GG:
      case PERIODIC_PRIM_LS:
      case PERIODIC_PRIM_ULS:
      case PERIODIC_PRIM_GG_R:
      case PERIODIC_PRIM_LS_R:
      case PERIODIC_PRIM_ULS_R:
      case PERIODIC_LIM_PRIM_1:
      case PERIODIC_LIM_PRIM_2:
        return nodes->GetPrimitive();
        break;
      default:
        return nodes->GetSolution();
        break;
    }
  }

  su2activematrix& selectLimiter(CVariable* nodes, unsigned short commType) {
    switch(commType) {
      case PERIODIC_LIM_PRIM_1:
      case PERIODIC_LIM_PRIM_2:
        return nodes->GetLimiter_Primitive();
        break;
      default:
        return nodes->GetLimiter();
        break;
    }
  }
}

void CSolver::InitiatePeriodicComms(CGeometry *geometry,
                                    const CConfig *config,
                                    unsigned short val_periodic_index,
                                    unsigned short commType) {

  /*--- Check for dummy communication. ---*/

  if (commType == PERIODIC_NONE) return;

  if (rotate_periodic && config->GetNEMOProblem()) {
    SU2_MPI::Error("The NEMO solvers do not support rotational periodicity yet.", CURRENT_FUNCTION);
  }

  /*--- Local variables ---*/

  bool boundary_i, boundary_j;
  bool weighted = true;

  unsigned short iVar, jVar, iDim;
  unsigned short nNeighbor       = 0;
  unsigned short COUNT_PER_POINT = 0;
  unsigned short MPI_TYPE        = 0;
  unsigned short ICOUNT          = nVar;
  unsigned short JCOUNT          = nVar;

  int iMessage, iSend, nSend;

  unsigned long iPoint, msg_offset, buf_offset, iPeriodic;

  su2double *Diff      = new su2double[nVar];
  su2double *Und_Lapl  = new su2double[nVar];
  su2double *Sol_Min   = new su2double[nPrimVarGrad];
  su2double *Sol_Max   = new su2double[nPrimVarGrad];
  su2double *rotPrim_i = new su2double[nPrimVar];
  su2double *rotPrim_j = new su2double[nPrimVar];

  su2double Sensor_i = 0.0, Sensor_j = 0.0, Pressure_i, Pressure_j;
  const su2double *Coord_i, *Coord_j;
  su2double r11, r12, r13, r22, r23_a, r23_b, r33, weight;
  const su2double *center, *angles, *trans;
  su2double rotMatrix2D[2][2] = {{1.0,0.0},{0.0,1.0}};
  su2double rotMatrix3D[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
  su2double rotCoord_i[3] = {0.0}, rotCoord_j[3] = {0.0};
  su2double translation[3] = {0.0}, distance[3] = {0.0};
  const su2double zeros[3] = {0.0};
  su2activematrix Cvector;

  auto Rotate = [&](const su2double* origin, const su2double* direction, su2double* rotated) {
    if(nDim==2) GeometryToolbox::Rotate(rotMatrix2D, origin, direction, rotated);
    else GeometryToolbox::Rotate(rotMatrix3D, origin, direction, rotated);
  };

  string Marker_Tag;

  /*--- Set the size of the data packet and type depending on quantity. ---*/

  GetPeriodicCommCountAndType(config, commType, COUNT_PER_POINT, MPI_TYPE, ICOUNT, JCOUNT);

  /*--- Allocate buffers for matrices that need rotation. ---*/

  su2activematrix jacBlock(ICOUNT,JCOUNT);
  su2activematrix rotBlock(ICOUNT,JCOUNT);

  /*--- Check to make sure we have created a large enough buffer
   for these comms during preprocessing. It will be reallocated whenever
   we find a larger count per point than currently exists. After the
   first cycle of comms, this should be inactive. ---*/

  geometry->AllocatePeriodicComms(COUNT_PER_POINT);

  /*--- Set some local pointers to make access simpler. ---*/

  su2double *bufDSend = geometry->bufD_PeriodicSend;

  unsigned short *bufSSend = geometry->bufS_PeriodicSend;

  /*--- Handle the different types of gradient and limiter. ---*/

  auto& gradient = PeriodicCommHelpers::selectGradient(base_nodes, commType);
  auto& limiter = PeriodicCommHelpers::selectLimiter(base_nodes, commType);
  auto& field = PeriodicCommHelpers::selectField(base_nodes, commType);

  /*--- Load the specified quantity from the solver into the generic
   communication buffer in the geometry class. ---*/

  if (geometry->nPeriodicSend > 0) {

    /*--- Post all non-blocking recvs first before sends. ---*/

    geometry->PostPeriodicRecvs(geometry, config, MPI_TYPE, COUNT_PER_POINT);

    for (iMessage = 0; iMessage < geometry->nPeriodicSend; iMessage++) {

      /*--- Get the offset in the buffer for the start of this message. ---*/

      msg_offset = geometry->nPoint_PeriodicSend[iMessage];

      /*--- Get the number of periodic points we need to
       communicate on the current periodic marker. ---*/

      nSend = (geometry->nPoint_PeriodicSend[iMessage+1] -
               geometry->nPoint_PeriodicSend[iMessage]);

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (iSend = 0; iSend < nSend; iSend++) {

        /*--- Get the local index for this communicated data. We need
         both the node and periodic face index (for rotations). ---*/

        iPoint    = geometry->Local_Point_PeriodicSend[msg_offset  + iSend];
        iPeriodic = geometry->Local_Marker_PeriodicSend[msg_offset + iSend];

        /*--- Retrieve the supplied periodic information. ---*/

        Marker_Tag = config->GetMarker_All_TagBound(iPeriodic);
        center     = config->GetPeriodicRotCenter(Marker_Tag);
        angles     = config->GetPeriodicRotAngles(Marker_Tag);
        trans      = config->GetPeriodicTranslation(Marker_Tag);

        /*--- Store (center+trans) as it is constant and will be added. ---*/

        translation[0] = center[0] + trans[0];
        translation[1] = center[1] + trans[1];
        translation[2] = center[2] + trans[2];

        /*--- Store angles separately for clarity. Compute sines/cosines. ---*/

        su2double Theta = angles[0];
        su2double Phi = angles[1];
        su2double Psi = angles[2];

        /*--- Compute the rotation matrix. Note that the implicit
         ordering is rotation about the x-axis, y-axis, then z-axis. ---*/

        if (nDim==2) {
          GeometryToolbox::RotationMatrix(Psi, rotMatrix2D);
        } else {
          GeometryToolbox::RotationMatrix(Theta, Phi, Psi, rotMatrix3D);
        }

        /*--- Compute the offset in the recv buffer for this point. ---*/

        buf_offset = (msg_offset + iSend)*COUNT_PER_POINT;

        /*--- Load the send buffers depending on the particular value
         that has been requested for communication. ---*/

        switch (commType) {

          case PERIODIC_VOLUME:

            /*--- Load the volume of the current periodic CV so that
             we can accumulate the total control volume size on all
             periodic faces. ---*/

            bufDSend[buf_offset] = geometry->nodes->GetVolume(iPoint) +
            geometry->nodes->GetPeriodicVolume(iPoint);

            break;

          case PERIODIC_NEIGHBORS:

            nNeighbor = 0;
            for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {

              /*--- Check if this neighbor lies on the periodic face so
               that we avoid double counting neighbors on both sides. If
               not, increment the count of neighbors for the donor. ---*/

              if (!geometry->nodes->GetPeriodicBoundary(jPoint))
                nNeighbor++;
            }

            /*--- Store the number of neighbors in bufffer. ---*/

            bufSSend[buf_offset] = nNeighbor;

            break;

          case PERIODIC_RESIDUAL:

            /*--- Communicate the residual from our partial control
             volume to the other side of the periodic face. ---*/

            for (iVar = 0; iVar < nVar; iVar++) {
              bufDSend[buf_offset+iVar] = LinSysRes(iPoint, iVar);
            }

            /*--- Rotate the momentum components of the residual array. ---*/

            if (rotate_periodic) {
              Rotate(zeros, &LinSysRes(iPoint,1), &bufDSend[buf_offset+1]);
            }
            buf_offset += nVar;

            /*--- Load the time step for the current point. ---*/

            bufDSend[buf_offset] = base_nodes->GetDelta_Time(iPoint);
            buf_offset++;

            /*--- For implicit calculations, we will communicate the
             contributions to the Jacobian block diagonal, i.e., the
             impact of the point upon itself, J_ii. ---*/

            if (implicit_periodic) {

              for (iVar = 0; iVar < nVar; iVar++) {
                for (jVar = 0; jVar < nVar; jVar++) {
                  jacBlock[iVar][jVar] = Jacobian.GetBlock(iPoint, iPoint, iVar, jVar);
                }
              }

              /*--- Rotate the momentum columns of the Jacobian. ---*/

              if (rotate_periodic) {
                for (iVar = 0; iVar < nVar; iVar++) {
                  if (nDim == 2) {
                    jacBlock[1][iVar] = (rotMatrix2D[0][0]*Jacobian.GetBlock(iPoint, iPoint, 1, iVar) +
                                         rotMatrix2D[0][1]*Jacobian.GetBlock(iPoint, iPoint, 2, iVar));
                    jacBlock[2][iVar] = (rotMatrix2D[1][0]*Jacobian.GetBlock(iPoint, iPoint, 1, iVar) +
                                         rotMatrix2D[1][1]*Jacobian.GetBlock(iPoint, iPoint, 2, iVar));
                  } else {

                    jacBlock[1][iVar] = (rotMatrix3D[0][0]*Jacobian.GetBlock(iPoint, iPoint, 1, iVar) +
                                         rotMatrix3D[0][1]*Jacobian.GetBlock(iPoint, iPoint, 2, iVar) +
                                         rotMatrix3D[0][2]*Jacobian.GetBlock(iPoint, iPoint, 3, iVar));
                    jacBlock[2][iVar] = (rotMatrix3D[1][0]*Jacobian.GetBlock(iPoint, iPoint, 1, iVar) +
                                         rotMatrix3D[1][1]*Jacobian.GetBlock(iPoint, iPoint, 2, iVar) +
                                         rotMatrix3D[1][2]*Jacobian.GetBlock(iPoint, iPoint, 3, iVar));
                    jacBlock[3][iVar] = (rotMatrix3D[2][0]*Jacobian.GetBlock(iPoint, iPoint, 1, iVar) +
                                         rotMatrix3D[2][1]*Jacobian.GetBlock(iPoint, iPoint, 2, iVar) +
                                         rotMatrix3D[2][2]*Jacobian.GetBlock(iPoint, iPoint, 3, iVar));
                  }
                }
              }

              /*--- Load the Jacobian terms into the buffer for sending. ---*/

              for (iVar = 0; iVar < nVar; iVar++) {
                for (jVar = 0; jVar < nVar; jVar++) {
                  bufDSend[buf_offset] = jacBlock[iVar][jVar];
                  buf_offset++;
                }
              }
            }

            break;

          case PERIODIC_IMPLICIT:

            /*--- Communicate the solution from our master set of periodic
             nodes (from the linear solver perspective) to the passive
             periodic nodes on the matching face. This is done at the
             end of the iteration to synchronize the solution after the
             linear solve. ---*/

            for (iVar = 0; iVar < nVar; iVar++) {
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution(iPoint, iVar);
            }

            /*--- Rotate the momentum components of the solution array. ---*/

            if (rotate_periodic) {
              Rotate(zeros, &base_nodes->GetSolution(iPoint)[1], &bufDSend[buf_offset+1]);
            }

            break;

          case PERIODIC_LAPLACIAN:

            /*--- For JST, the undivided Laplacian must be computed
             consistently by using the complete control volume info
             from both sides of the periodic face. ---*/

            for (iVar = 0; iVar < nVar; iVar++)
              Und_Lapl[iVar] = 0.0;

            for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {

              /*--- Avoid periodic boundary points so that we do not
               duplicate edges on both sides of the periodic BC. ---*/

              if (!geometry->nodes->GetPeriodicBoundary(jPoint)) {

                /*--- Solution differences ---*/

                for (iVar = 0; iVar < nVar; iVar++)
                Diff[iVar] = (base_nodes->GetSolution(iPoint, iVar) -
                              base_nodes->GetSolution(jPoint,iVar));

                /*--- Correction for compressible flows (use enthalpy) ---*/

                if (!(config->GetKind_Regime() == ENUM_REGIME::INCOMPRESSIBLE)) {
                  Pressure_i   = base_nodes->GetPressure(iPoint);
                  Pressure_j   = base_nodes->GetPressure(jPoint);
                  Diff[nVar-1] = ((base_nodes->GetSolution(iPoint,nVar-1) + Pressure_i) -
                                  (base_nodes->GetSolution(jPoint,nVar-1) + Pressure_j));
                }

                boundary_i = geometry->nodes->GetPhysicalBoundary(iPoint);
                boundary_j = geometry->nodes->GetPhysicalBoundary(jPoint);

                /*--- Both points inside the domain, or both in the boundary ---*/
                /*--- iPoint inside the domain, jPoint on the boundary ---*/

                if (!(boundary_i && !boundary_j)) {
                  if (geometry->nodes->GetDomain(iPoint)){
                    for (iVar = 0; iVar< nVar; iVar++)
                    Und_Lapl[iVar] -= Diff[iVar];
                  }
                }
              }
            }

            /*--- Store the components to be communicated in the buffer. ---*/

            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = Und_Lapl[iVar];

            /*--- Rotate the momentum components of the Laplacian. ---*/

            if (rotate_periodic) {
              Rotate(zeros, &Und_Lapl[1], &bufDSend[buf_offset+1]);
            }

            break;

          case PERIODIC_MAX_EIG:

            /*--- Simple summation of eig calc on both periodic faces. ---*/

            bufDSend[buf_offset] = base_nodes->GetLambda(iPoint);

            break;

          case PERIODIC_SENSOR:

            /*--- For the centered schemes, the sensor must be computed
             consistently using info from the entire control volume
             on both sides of the periodic face. ---*/

            Sensor_i = 0.0; Sensor_j = 0.0;
            for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {

              /*--- Avoid halos and boundary points so that we don't
               duplicate edges on both sides of the periodic BC. ---*/

              if (!geometry->nodes->GetPeriodicBoundary(jPoint)) {

                /*--- Use density instead of pressure for incomp. flows. ---*/

                if ((config->GetKind_Regime() == ENUM_REGIME::INCOMPRESSIBLE)) {
                  Pressure_i = base_nodes->GetDensity(iPoint);
                  Pressure_j = base_nodes->GetDensity(jPoint);
                } else {
                  Pressure_i = base_nodes->GetPressure(iPoint);
                  Pressure_j = base_nodes->GetPressure(jPoint);
                }

                boundary_i = geometry->nodes->GetPhysicalBoundary(iPoint);
                boundary_j = geometry->nodes->GetPhysicalBoundary(jPoint);

                /*--- Both points inside domain, or both on boundary ---*/
                /*--- iPoint inside the domain, jPoint on the boundary ---*/

                if (!(boundary_i && !boundary_j)) {
                  if (geometry->nodes->GetDomain(iPoint)) {
                    Sensor_i += (Pressure_j - Pressure_i);
                    Sensor_j += (Pressure_i + Pressure_j);
                  }
                }

              }
            }

            /*--- Store the sensor increments to buffer. After summing
             all contributions, these will be divided. ---*/

            bufDSend[buf_offset] = Sensor_i;
            buf_offset++;
            bufDSend[buf_offset] = Sensor_j;

            break;

          case PERIODIC_SOL_GG:
          case PERIODIC_SOL_GG_R:
          case PERIODIC_PRIM_GG:
          case PERIODIC_PRIM_GG_R:

            /*--- Access and rotate the partial G-G gradient. These will be
             summed on both sides of the periodic faces before dividing
             by the volume to complete the Green-Gauss gradient calc. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              for (iDim = 0; iDim < nDim; iDim++) {
                jacBlock[iVar][iDim] = gradient(iPoint, iVar, iDim);
              }
            }

            /*--- Rotate the gradients in x,y,z space for all variables. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              Rotate(zeros, jacBlock[iVar], rotBlock[iVar]);
            }

            /*--- Rotate the vector components of the solution. ---*/

            if (rotate_periodic) {
              for (iDim = 0; iDim < nDim; iDim++) {
                su2double d_diDim[3] = {0.0};
                for (iVar = 1; iVar < 1+nDim; ++iVar) {
                  d_diDim[iVar-1] = rotBlock(iVar, iDim);
                }
                su2double rotated[3] = {0.0};
                Rotate(zeros, d_diDim, rotated);
                for (iVar = 1; iVar < 1+nDim; ++iVar) {
                  rotBlock(iVar, iDim) = rotated[iVar-1];
                }
              }
            }

            /*--- Store the partial gradient in the buffer. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              for (iDim = 0; iDim < nDim; iDim++) {
                bufDSend[buf_offset+iVar*nDim+iDim] = rotBlock[iVar][iDim];
              }
            }

            break;

          case PERIODIC_SOL_LS: case PERIODIC_SOL_ULS:
          case PERIODIC_SOL_LS_R: case PERIODIC_SOL_ULS_R:
          case PERIODIC_PRIM_LS: case PERIODIC_PRIM_ULS:
          case PERIODIC_PRIM_LS_R: case PERIODIC_PRIM_ULS_R:

            /*--- For L-S gradient calculations with rotational periodicity,
             we will need to rotate the x,y,z components. To make the process
             easier, we choose to rotate the initial periodic point and their
             neighbor points into their location on the donor marker before
             computing the terms that we need to communicate. ---*/

            /*--- Set a flag for unweighted or weighted least-squares. ---*/

            switch(commType) {
              case PERIODIC_SOL_ULS:
              case PERIODIC_SOL_ULS_R:
              case PERIODIC_PRIM_ULS:
              case PERIODIC_PRIM_ULS_R:
                weighted = false;
                break;
              default:
                weighted = true;
                break;
            }

            /*--- Get coordinates for the current point. ---*/

            Coord_i = geometry->nodes->GetCoord(iPoint);

            /*--- Get the position vector from rotation center to point. ---*/

            GeometryToolbox::Distance(nDim, Coord_i, center, distance);

            /*--- Compute transformed point coordinates. ---*/

            Rotate(translation, distance, rotCoord_i);

            /*--- Get conservative solution and rotate if necessary. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++)
              rotPrim_i[iVar] = field(iPoint, iVar);

            if (rotate_periodic) {
              Rotate(zeros, &field(iPoint,1), &rotPrim_i[1]);
            }

            /*--- Inizialization of variables ---*/

            Cvector.resize(ICOUNT,nDim) = su2double(0.0);

            r11 = 0.0;   r12 = 0.0;   r22 = 0.0;
            r13 = 0.0; r23_a = 0.0; r23_b = 0.0;  r33 = 0.0;

            for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {

              /*--- Avoid periodic boundary points so that we do not
               duplicate edges on both sides of the periodic BC. ---*/

              if (!geometry->nodes->GetPeriodicBoundary(jPoint)) {

                /*--- Get coordinates for the neighbor point. ---*/

                Coord_j = geometry->nodes->GetCoord(jPoint);

                /*--- Get the position vector from rotation center. ---*/

                GeometryToolbox::Distance(nDim, Coord_j, center, distance);

                /*--- Compute transformed point coordinates. ---*/

                Rotate(translation, distance, rotCoord_j);

                /*--- Get conservative solution and rotate if necessary. ---*/

                for (iVar = 0; iVar < ICOUNT; iVar++)
                  rotPrim_j[iVar] = field(jPoint,iVar);

                if (rotate_periodic) {
                  Rotate(zeros, &field(jPoint,1), &rotPrim_j[1]);
                }

                if (weighted) {
                  weight = GeometryToolbox::SquaredDistance(nDim, rotCoord_j, rotCoord_i);
                } else {
                  weight = 1.0;
                }

                /*--- Sumations for entries of upper triangular matrix R ---*/

                if (weight != 0.0) {

                  r11 += ((rotCoord_j[0]-rotCoord_i[0])*
                          (rotCoord_j[0]-rotCoord_i[0])/weight);
                  r12 += ((rotCoord_j[0]-rotCoord_i[0])*
                          (rotCoord_j[1]-rotCoord_i[1])/weight);
                  r22 += ((rotCoord_j[1]-rotCoord_i[1])*
                          (rotCoord_j[1]-rotCoord_i[1])/weight);

                  if (nDim == 3) {
                    r13   += ((rotCoord_j[0]-rotCoord_i[0])*
                              (rotCoord_j[2]-rotCoord_i[2])/weight);
                    r23_a += ((rotCoord_j[1]-rotCoord_i[1])*
                              (rotCoord_j[2]-rotCoord_i[2])/weight);
                    r23_b += ((rotCoord_j[0]-rotCoord_i[0])*
                              (rotCoord_j[2]-rotCoord_i[2])/weight);
                    r33   += ((rotCoord_j[2]-rotCoord_i[2])*
                              (rotCoord_j[2]-rotCoord_i[2])/weight);
                  }

                  /*--- Entries of c:= transpose(A)*b ---*/

                  for (iVar = 0; iVar < ICOUNT; iVar++)
                  for (iDim = 0; iDim < nDim; iDim++)
                  Cvector(iVar,iDim) += ((rotCoord_j[iDim]-rotCoord_i[iDim])*
                                          (rotPrim_j[iVar]-rotPrim_i[iVar])/weight);

                }
              }
            }

            /*--- We store and communicate the increments for the matching
             upper triangular matrix (weights) and the r.h.s. vector.
             These will be accumulated before completing the L-S gradient
             calculation for each periodic point. ---*/

            if (nDim == 2) {
              bufDSend[buf_offset] = r11;   buf_offset++;
              bufDSend[buf_offset] = r12;   buf_offset++;
              bufDSend[buf_offset] = 0.0;   buf_offset++;
              bufDSend[buf_offset] = r22;   buf_offset++;
            }
            if (nDim == 3) {
              bufDSend[buf_offset] = r11;   buf_offset++;
              bufDSend[buf_offset] = r12;   buf_offset++;
              bufDSend[buf_offset] = r13;   buf_offset++;

              bufDSend[buf_offset] = 0.0;   buf_offset++;
              bufDSend[buf_offset] = r22;   buf_offset++;
              bufDSend[buf_offset] = r23_a; buf_offset++;

              bufDSend[buf_offset] = 0.0;   buf_offset++;
              bufDSend[buf_offset] = r23_b; buf_offset++;
              bufDSend[buf_offset] = r33;   buf_offset++;
            }

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              for (iDim = 0; iDim < nDim; iDim++) {
                bufDSend[buf_offset] = Cvector(iVar,iDim);
                buf_offset++;
              }
            }

            break;

          case PERIODIC_LIM_PRIM_1:
          case PERIODIC_LIM_SOL_1:

            /*--- The first phase of the periodic limiter calculation
             ensures that the proper min and max of the solution are found
             among all nodes adjacent to periodic faces. ---*/

            /*--- We send the min and max over "our" neighbours. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              Sol_Min[iVar] = base_nodes->GetSolution_Min()(iPoint, iVar);
              Sol_Max[iVar] = base_nodes->GetSolution_Max()(iPoint, iVar);
            }

            for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {
              for (iVar = 0; iVar < ICOUNT; iVar++) {
                Sol_Min[iVar] = min(Sol_Min[iVar], field(jPoint, iVar));
                Sol_Max[iVar] = max(Sol_Max[iVar], field(jPoint, iVar));
              }
            }

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              bufDSend[buf_offset+iVar]        = Sol_Min[iVar];
              bufDSend[buf_offset+ICOUNT+iVar] = Sol_Max[iVar];
            }

            /*--- Rotate the momentum components of the min/max. ---*/

            if (rotate_periodic) {
              Rotate(zeros, &Sol_Min[1], &bufDSend[buf_offset+1]);
              Rotate(zeros, &Sol_Max[1], &bufDSend[buf_offset+ICOUNT+1]);
            }

            break;

          case PERIODIC_LIM_PRIM_2:
          case PERIODIC_LIM_SOL_2:

            /*--- The second phase of the periodic limiter calculation
             ensures that the correct minimum value of the limiter is
             found for a node on a periodic face and stores it. ---*/

            for (iVar = 0; iVar < ICOUNT; iVar++) {
              bufDSend[buf_offset+iVar] = limiter(iPoint, iVar);
            }

            if (rotate_periodic) {
              Rotate(zeros, &limiter(iPoint,1), &bufDSend[buf_offset+1]);
            }

            break;

          default:
            SU2_MPI::Error("Unrecognized quantity for periodic communication.",
                           CURRENT_FUNCTION);
            break;
        }
      }
      END_SU2_OMP_FOR

      /*--- Launch the point-to-point MPI send for this message. ---*/

      geometry->PostPeriodicSends(geometry, config, MPI_TYPE, COUNT_PER_POINT, iMessage);

    }
  }

  delete [] Diff;
  delete [] Und_Lapl;
  delete [] Sol_Min;
  delete [] Sol_Max;
  delete [] rotPrim_i;
  delete [] rotPrim_j;

}

void CSolver::CompletePeriodicComms(CGeometry *geometry,
                                    const CConfig *config,
                                    unsigned short val_periodic_index,
                                    unsigned short commType) {

  /*--- Check for dummy communication. ---*/

  if (commType == PERIODIC_NONE) return;

  /*--- Set the size of the data packet and type depending on quantity. ---*/

  unsigned short COUNT_PER_POINT = 0, MPI_TYPE = 0, ICOUNT = 0, JCOUNT = 0;
  GetPeriodicCommCountAndType(config, commType, COUNT_PER_POINT, MPI_TYPE, ICOUNT, JCOUNT);

  /*--- Local variables ---*/

  unsigned short nPeriodic = config->GetnMarker_Periodic();
  unsigned short iDim, jDim, iVar, jVar, iPeriodic, nNeighbor;

  unsigned long iPoint, iRecv, nRecv, msg_offset, buf_offset, total_index;

  int source, iMessage, jRecv;

  /*--- Status is global so all threads can see the result of Waitany. ---*/
  static SU2_MPI::Status status;

  su2double *Diff = new su2double[nVar];

  su2double Time_Step, Volume;

  su2double **Jacobian_i = nullptr;
  if ((commType == PERIODIC_RESIDUAL) && implicit_periodic) {
    Jacobian_i = new su2double* [nVar];
    for (iVar = 0; iVar < nVar; iVar++)
      Jacobian_i[iVar] = new su2double [nVar];
  }

  /*--- Set some local pointers to make access simpler. ---*/

  const su2double *bufDRecv = geometry->bufD_PeriodicRecv;

  const unsigned short *bufSRecv = geometry->bufS_PeriodicRecv;

  /*--- Handle the different types of gradient and limiter. ---*/

  auto& gradient = PeriodicCommHelpers::selectGradient(base_nodes, commType);
  auto& limiter = PeriodicCommHelpers::selectLimiter(base_nodes, commType);

  /*--- Store the data that was communicated into the appropriate
   location within the local class data structures. ---*/

  if (geometry->nPeriodicRecv > 0) {

    for (iMessage = 0; iMessage < geometry->nPeriodicRecv; iMessage++) {

      /*--- For efficiency, recv the messages dynamically based on
       the order they arrive. ---*/

#ifdef HAVE_MPI
      /*--- Once we have recv'd a message, get the source rank. ---*/
      int ind;
      SU2_OMP_SAFE_GLOBAL_ACCESS(SU2_MPI::Waitany(geometry->nPeriodicRecv, geometry->req_PeriodicRecv, &ind, &status);)
      source = status.MPI_SOURCE;
#else
      /*--- For serial calculations, we know the rank. ---*/
      source = rank;
      SU2_OMP_BARRIER
#endif

      /*--- We know the offsets based on the source rank. ---*/

      jRecv = geometry->PeriodicRecv2Neighbor[source];

      /*--- Get the offset in the buffer for the start of this message. ---*/

      msg_offset = geometry->nPoint_PeriodicRecv[jRecv];

      /*--- Get the number of packets to be received in this message. ---*/

      nRecv = (geometry->nPoint_PeriodicRecv[jRecv+1] -
               geometry->nPoint_PeriodicRecv[jRecv]);

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (iRecv = 0; iRecv < nRecv; iRecv++) {

        /*--- Get the local index for this communicated data. ---*/

        iPoint    = geometry->Local_Point_PeriodicRecv[msg_offset  + iRecv];
        iPeriodic = geometry->Local_Marker_PeriodicRecv[msg_offset + iRecv];

        /*--- While all periodic face data was accumulated, we only store
         the values for the current pair of periodic faces. This is slightly
         inefficient when we have multiple pairs of periodic faces, but
         it simplifies the communications. ---*/

        if ((iPeriodic == val_periodic_index) ||
            (iPeriodic == val_periodic_index + nPeriodic/2)) {

          /*--- Compute the offset in the recv buffer for this point. ---*/

          buf_offset = (msg_offset + iRecv)*COUNT_PER_POINT;

          /*--- Store the data correctly depending on the quantity. ---*/

          switch (commType) {

            case PERIODIC_VOLUME:

              /*--- The periodic points need to keep track of their
               total volume spread across the periodic faces. ---*/

              Volume = (bufDRecv[buf_offset] +
                        geometry->nodes->GetPeriodicVolume(iPoint));
              geometry->nodes->SetPeriodicVolume(iPoint, Volume);

              break;

            case PERIODIC_NEIGHBORS:

              /*--- Store the extra neighbors on the periodic face. ---*/

              nNeighbor = (geometry->nodes->GetnNeighbor(iPoint) +
                           bufSRecv[buf_offset]);
              geometry->nodes->SetnNeighbor(iPoint, nNeighbor);

              break;

            case PERIODIC_RESIDUAL:

              /*--- Add contributions to total residual. ---*/

              LinSysRes.AddBlock(iPoint, &bufDRecv[buf_offset]);
              buf_offset += nVar;

              /*--- Check the computed time step against the donor
               value and keep the minimum in order to be conservative. ---*/

              Time_Step = base_nodes->GetDelta_Time(iPoint);
              if (bufDRecv[buf_offset] < Time_Step)
                base_nodes->SetDelta_Time(iPoint,bufDRecv[buf_offset]);
              buf_offset++;

              /*--- For implicit integration, we choose the first
               periodic face of each pair to be the master/owner of
               the solution for the linear system while fixing the
               solution at the matching face during the solve. Here,
               we remove the Jacobian and residual contributions from
               the passive face such that it does not participate in
               the linear solve. ---*/

              if (implicit_periodic) {

                for (iVar = 0; iVar < nVar; iVar++) {
                  for (jVar = 0; jVar < nVar; jVar++) {
                    Jacobian_i[iVar][jVar] = bufDRecv[buf_offset];
                    buf_offset++;
                  }
                }

                Jacobian.AddBlock2Diag(iPoint, Jacobian_i);

                if (iPeriodic == val_periodic_index + nPeriodic/2) {
                  for (iVar = 0; iVar < nVar; iVar++) {
                    LinSysRes(iPoint, iVar) = 0.0;
                    total_index = iPoint*nVar+iVar;
                    Jacobian.DeleteValsRowi(total_index);
                  }
                }

              }

              break;

            case PERIODIC_IMPLICIT:

              /*--- For implicit integration, we choose the first
               periodic face of each pair to be the master/owner of
               the solution for the linear system while fixing the
               solution at the matching face during the solve. Here,
               we are updating the solution at the passive nodes
               using the new solution from the master. ---*/

              if ((implicit_periodic) &&
                  (iPeriodic == val_periodic_index + nPeriodic/2)) {

                /*--- Directly set the solution on the passive periodic
                 face that is provided from the master. ---*/

                for (iVar = 0; iVar < nVar; iVar++) {
                  base_nodes->SetSolution(iPoint, iVar, bufDRecv[buf_offset]);
                  base_nodes->SetSolution_Old(iPoint, iVar, bufDRecv[buf_offset]);
                  buf_offset++;
                }

              }

              break;

            case PERIODIC_LAPLACIAN:

              /*--- Adjust the undivided Laplacian. The accumulation was
               with a subtraction before communicating, so now just add. ---*/

              for (iVar = 0; iVar < nVar; iVar++)
                base_nodes->AddUnd_Lapl(iPoint, iVar, bufDRecv[buf_offset+iVar]);

              break;

            case PERIODIC_MAX_EIG:

              /*--- Simple accumulation of the max eig on periodic faces. ---*/

              base_nodes->AddLambda(iPoint,bufDRecv[buf_offset]);

              break;

            case PERIODIC_SENSOR:

              /*--- Simple accumulation of the sensors on periodic faces. ---*/

              iPoint_UndLapl[iPoint] += bufDRecv[buf_offset]; buf_offset++;
              jPoint_UndLapl[iPoint] += bufDRecv[buf_offset];

              break;

            case PERIODIC_SOL_GG:
            case PERIODIC_SOL_GG_R:
            case PERIODIC_PRIM_GG:
            case PERIODIC_PRIM_GG_R:

              /*--- For G-G, we accumulate partial gradients then compute
               the final value using the entire volume of the periodic cell. ---*/

              for (iVar = 0; iVar < ICOUNT; iVar++)
                for (iDim = 0; iDim < nDim; iDim++)
                  gradient(iPoint, iVar, iDim) += bufDRecv[buf_offset+iVar*nDim+iDim];

              break;

            case PERIODIC_SOL_LS: case PERIODIC_SOL_ULS:
            case PERIODIC_SOL_LS_R: case PERIODIC_SOL_ULS_R:
            case PERIODIC_PRIM_LS: case PERIODIC_PRIM_ULS:
            case PERIODIC_PRIM_LS_R: case PERIODIC_PRIM_ULS_R:

              /*--- For L-S, we build the upper triangular matrix and the
               r.h.s. vector by accumulating from all periodic partial
               control volumes. ---*/

              for (iDim = 0; iDim < nDim; iDim++) {
                for (jDim = 0; jDim < nDim; jDim++) {
                  base_nodes->AddRmatrix(iPoint, iDim,jDim,bufDRecv[buf_offset]);
                  buf_offset++;
                }
              }
              for (iVar = 0; iVar < ICOUNT; iVar++) {
                for (iDim = 0; iDim < nDim; iDim++) {
                  gradient(iPoint, iVar, iDim) += bufDRecv[buf_offset];
                  buf_offset++;
                }
              }

              break;

            case PERIODIC_LIM_PRIM_1:
            case PERIODIC_LIM_SOL_1:

              /*--- Update solution min/max with min/max between "us" and
               the periodic match plus its neighbors, computation will need to
               be concluded on "our" side to account for "our" neighbors. ---*/

              for (iVar = 0; iVar < ICOUNT; iVar++) {

                /*--- Solution minimum. ---*/

                su2double Solution_Min = min(base_nodes->GetSolution_Min()(iPoint, iVar),
                                             bufDRecv[buf_offset+iVar]);
                base_nodes->GetSolution_Min()(iPoint, iVar) = Solution_Min;

                /*--- Solution maximum. ---*/

                su2double Solution_Max = max(base_nodes->GetSolution_Max()(iPoint, iVar),
                                             bufDRecv[buf_offset+ICOUNT+iVar]);
                base_nodes->GetSolution_Max()(iPoint, iVar) = Solution_Max;
              }

              break;

            case PERIODIC_LIM_PRIM_2:
            case PERIODIC_LIM_SOL_2:

              /*--- Check the min values found on the matching periodic
               faces for the limiter, and store the proper min value. ---*/

              for (iVar = 0; iVar < ICOUNT; iVar++)
                limiter(iPoint, iVar) = min(limiter(iPoint, iVar), bufDRecv[buf_offset+iVar]);

              break;

            default:

              SU2_MPI::Error("Unrecognized quantity for periodic communication.",
                             CURRENT_FUNCTION);
              break;

          }
        }
      }
      END_SU2_OMP_FOR
    }

    /*--- Verify that all non-blocking point-to-point sends have finished.
     Note that this should be satisfied, as we have received all of the
     data in the loop above at this point. ---*/

#ifdef HAVE_MPI
    SU2_OMP_SAFE_GLOBAL_ACCESS(SU2_MPI::Waitall(geometry->nPeriodicSend, geometry->req_PeriodicSend, MPI_STATUS_IGNORE);)
#endif
  }

  delete [] Diff;

  if (Jacobian_i)
    for (iVar = 0; iVar < nVar; iVar++)
      delete [] Jacobian_i[iVar];
  delete [] Jacobian_i;

}

void CSolver::GetCommCountAndType(const CConfig* config,
                                  unsigned short commType,
                                  unsigned short &COUNT_PER_POINT,
                                  unsigned short &MPI_TYPE) const {
  switch (commType) {
    case SOLUTION:
    case SOLUTION_OLD:
    case UNDIVIDED_LAPLACIAN:
    case SOLUTION_LIMITER:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case MAX_EIGENVALUE:
    case SENSOR:
      COUNT_PER_POINT  = 1;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case SOLUTION_GRADIENT:
    case SOLUTION_GRAD_REC:
      COUNT_PER_POINT  = nVar*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PRIMITIVE_GRADIENT:
    case PRIMITIVE_GRAD_REC:
      COUNT_PER_POINT  = nPrimVarGrad*nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case PRIMITIVE_LIMITER:
      COUNT_PER_POINT  = nPrimVarGrad;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case SOLUTION_EDDY:
      COUNT_PER_POINT  = nVar+1;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case SOLUTION_FEA:
      if (config->GetTime_Domain())
        COUNT_PER_POINT  = nVar*3;
      else
        COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case AUXVAR_GRADIENT:
      COUNT_PER_POINT  = nDim*base_nodes->GetnAuxVar();
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case MESH_DISPLACEMENTS:
      COUNT_PER_POINT  = nDim;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case SOLUTION_TIME_N:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    case SOLUTION_TIME_N1:
      COUNT_PER_POINT  = nVar;
      MPI_TYPE         = COMM_TYPE_DOUBLE;
      break;
    default:
      SU2_MPI::Error("Unrecognized quantity for point-to-point MPI comms.",
                     CURRENT_FUNCTION);
      break;
  }
}

namespace CommHelpers {
  CVectorOfMatrix& selectGradient(CVariable* nodes, unsigned short commType) {
    switch(commType) {
      case SOLUTION_GRAD_REC: return nodes->GetGradient_Reconstruction();
      case PRIMITIVE_GRADIENT: return nodes->GetGradient_Primitive();
      case PRIMITIVE_GRAD_REC: return nodes->GetGradient_Reconstruction();
      case AUXVAR_GRADIENT: return nodes->GetAuxVarGradient();
      default: return nodes->GetGradient();
    }
  }

  su2activematrix& selectLimiter(CVariable* nodes, unsigned short commType) {
    if (commType == PRIMITIVE_LIMITER) return nodes->GetLimiter_Primitive();
    return nodes->GetLimiter();
  }
}

void CSolver::InitiateComms(CGeometry *geometry,
                            const CConfig *config,
                            unsigned short commType) {

  /*--- Local variables ---*/

  unsigned short iVar, iDim;
  unsigned short COUNT_PER_POINT = 0;
  unsigned short MPI_TYPE        = 0;

  unsigned long iPoint, msg_offset, buf_offset;

  int iMessage, iSend, nSend;

  /*--- Set the size of the data packet and type depending on quantity. ---*/

  GetCommCountAndType(config, commType, COUNT_PER_POINT, MPI_TYPE);

  /*--- Check to make sure we have created a large enough buffer
   for these comms during preprocessing. This is only for the su2double
   buffer. It will be reallocated whenever we find a larger count
   per point. After the first cycle of comms, this should be inactive. ---*/

  geometry->AllocateP2PComms(COUNT_PER_POINT);

  /*--- Set some local pointers to make access simpler. ---*/

  su2double *bufDSend = geometry->bufD_P2PSend;

  /*--- Handle the different types of gradient and limiter. ---*/

  const auto nVarGrad = COUNT_PER_POINT / nDim;
  auto& gradient = CommHelpers::selectGradient(base_nodes, commType);
  auto& limiter = CommHelpers::selectLimiter(base_nodes, commType);

  /*--- Load the specified quantity from the solver into the generic
   communication buffer in the geometry class. ---*/

  if (geometry->nP2PSend > 0) {

    /*--- Post all non-blocking recvs first before sends. ---*/

    geometry->PostP2PRecvs(geometry, config, MPI_TYPE, COUNT_PER_POINT, false);

    for (iMessage = 0; iMessage < geometry->nP2PSend; iMessage++) {

      /*--- Get the offset in the buffer for the start of this message. ---*/

      msg_offset = geometry->nPoint_P2PSend[iMessage];

      /*--- Total count can include multiple pieces of data per element. ---*/

      nSend = (geometry->nPoint_P2PSend[iMessage+1] -
               geometry->nPoint_P2PSend[iMessage]);

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (iSend = 0; iSend < nSend; iSend++) {

        /*--- Get the local index for this communicated data. ---*/

        iPoint = geometry->Local_Point_P2PSend[msg_offset + iSend];

        /*--- Compute the offset in the recv buffer for this point. ---*/

        buf_offset = (msg_offset + iSend)*COUNT_PER_POINT;

        switch (commType) {
          case SOLUTION:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution(iPoint, iVar);
            break;
          case SOLUTION_OLD:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution_Old(iPoint, iVar);
            break;
          case SOLUTION_EDDY:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution(iPoint, iVar);
            bufDSend[buf_offset+nVar]   = base_nodes->GetmuT(iPoint);
            break;
          case UNDIVIDED_LAPLACIAN:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetUndivided_Laplacian(iPoint, iVar);
            break;
          case SOLUTION_LIMITER:
          case PRIMITIVE_LIMITER:
            for (iVar = 0; iVar < COUNT_PER_POINT; iVar++)
              bufDSend[buf_offset+iVar] = limiter(iPoint, iVar);
            break;
          case MAX_EIGENVALUE:
            bufDSend[buf_offset] = base_nodes->GetLambda(iPoint);
            break;
          case SENSOR:
            bufDSend[buf_offset] = base_nodes->GetSensor(iPoint);
            break;
          case SOLUTION_GRADIENT:
          case PRIMITIVE_GRADIENT:
          case SOLUTION_GRAD_REC:
          case PRIMITIVE_GRAD_REC:
          case AUXVAR_GRADIENT:
            for (iVar = 0; iVar < nVarGrad; iVar++)
              for (iDim = 0; iDim < nDim; iDim++)
                bufDSend[buf_offset+iVar*nDim+iDim] = gradient(iPoint, iVar, iDim);
            break;
          case SOLUTION_FEA:
            for (iVar = 0; iVar < nVar; iVar++) {
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution(iPoint, iVar);
              if (config->GetTime_Domain()) {
                bufDSend[buf_offset+nVar+iVar]   = base_nodes->GetSolution_Vel(iPoint, iVar);
                bufDSend[buf_offset+nVar*2+iVar] = base_nodes->GetSolution_Accel(iPoint, iVar);
              }
            }
            break;
          case MESH_DISPLACEMENTS:
            for (iDim = 0; iDim < nDim; iDim++)
              bufDSend[buf_offset+iDim] = base_nodes->GetBound_Disp(iPoint, iDim);
            break;
          case SOLUTION_TIME_N:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution_time_n(iPoint, iVar);
            break;
          case SOLUTION_TIME_N1:
            for (iVar = 0; iVar < nVar; iVar++)
              bufDSend[buf_offset+iVar] = base_nodes->GetSolution_time_n1(iPoint, iVar);
            break;
          default:
            SU2_MPI::Error("Unrecognized quantity for point-to-point MPI comms.",
                           CURRENT_FUNCTION);
            break;
        }
      }
      END_SU2_OMP_FOR

      /*--- Launch the point-to-point MPI send for this message. ---*/

      geometry->PostP2PSends(geometry, config, MPI_TYPE, COUNT_PER_POINT, iMessage, false);

    }
  }

}

void CSolver::CompleteComms(CGeometry *geometry,
                            const CConfig *config,
                            unsigned short commType) {

  /*--- Local variables ---*/

  unsigned short iDim, iVar;
  unsigned long iPoint, iRecv, nRecv, msg_offset, buf_offset;
  unsigned short COUNT_PER_POINT = 0;
  unsigned short MPI_TYPE = 0;

  int ind, source, iMessage, jRecv;

  /*--- Global status so all threads can see the result of Waitany. ---*/
  static SU2_MPI::Status status;

  /*--- Set the size of the data packet and type depending on quantity. ---*/

  GetCommCountAndType(config, commType, COUNT_PER_POINT, MPI_TYPE);

  /*--- Set some local pointers to make access simpler. ---*/

  const su2double *bufDRecv = geometry->bufD_P2PRecv;

  /*--- Handle the different types of gradient and limiter. ---*/

  const auto nVarGrad = COUNT_PER_POINT / nDim;
  auto& gradient = CommHelpers::selectGradient(base_nodes, commType);
  auto& limiter = CommHelpers::selectLimiter(base_nodes, commType);

  /*--- Store the data that was communicated into the appropriate
   location within the local class data structures. ---*/

  if (geometry->nP2PRecv > 0) {

    for (iMessage = 0; iMessage < geometry->nP2PRecv; iMessage++) {

      /*--- For efficiency, recv the messages dynamically based on
       the order they arrive. ---*/

      SU2_OMP_SAFE_GLOBAL_ACCESS(SU2_MPI::Waitany(geometry->nP2PRecv, geometry->req_P2PRecv, &ind, &status);)

      /*--- Once we have recv'd a message, get the source rank. ---*/

      source = status.MPI_SOURCE;

      /*--- We know the offsets based on the source rank. ---*/

      jRecv = geometry->P2PRecv2Neighbor[source];

      /*--- Get the offset in the buffer for the start of this message. ---*/

      msg_offset = geometry->nPoint_P2PRecv[jRecv];

      /*--- Get the number of packets to be received in this message. ---*/

      nRecv = (geometry->nPoint_P2PRecv[jRecv+1] -
               geometry->nPoint_P2PRecv[jRecv]);

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (iRecv = 0; iRecv < nRecv; iRecv++) {

        /*--- Get the local index for this communicated data. ---*/

        iPoint = geometry->Local_Point_P2PRecv[msg_offset + iRecv];

        /*--- Compute the offset in the recv buffer for this point. ---*/

        buf_offset = (msg_offset + iRecv)*COUNT_PER_POINT;

        /*--- Store the data correctly depending on the quantity. ---*/

        switch (commType) {
          case SOLUTION:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->SetSolution(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            break;
          case SOLUTION_OLD:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->SetSolution_Old(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            break;
          case SOLUTION_EDDY:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->SetSolution(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            base_nodes->SetmuT(iPoint,bufDRecv[buf_offset+nVar]);
            break;
          case UNDIVIDED_LAPLACIAN:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->SetUnd_Lapl(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            break;
          case SOLUTION_LIMITER:
          case PRIMITIVE_LIMITER:
            for (iVar = 0; iVar < COUNT_PER_POINT; iVar++)
              limiter(iPoint,iVar) = bufDRecv[buf_offset+iVar];
            break;
          case MAX_EIGENVALUE:
            base_nodes->SetLambda(iPoint,bufDRecv[buf_offset]);
            break;
          case SENSOR:
            base_nodes->SetSensor(iPoint,bufDRecv[buf_offset]);
            break;
          case SOLUTION_GRADIENT:
          case PRIMITIVE_GRADIENT:
          case SOLUTION_GRAD_REC:
          case PRIMITIVE_GRAD_REC:
          case AUXVAR_GRADIENT:
            for (iVar = 0; iVar < nVarGrad; iVar++)
              for (iDim = 0; iDim < nDim; iDim++)
                gradient(iPoint,iVar,iDim) = bufDRecv[buf_offset+iVar*nDim+iDim];
            break;
          case SOLUTION_FEA:
            for (iVar = 0; iVar < nVar; iVar++) {
              base_nodes->SetSolution(iPoint, iVar, bufDRecv[buf_offset+iVar]);
              if (config->GetTime_Domain()) {
                base_nodes->SetSolution_Vel(iPoint, iVar, bufDRecv[buf_offset+nVar+iVar]);
                base_nodes->SetSolution_Accel(iPoint, iVar, bufDRecv[buf_offset+nVar*2+iVar]);
              }
            }
            break;
          case MESH_DISPLACEMENTS:
            for (iDim = 0; iDim < nDim; iDim++)
              base_nodes->SetBound_Disp(iPoint, iDim, bufDRecv[buf_offset+iDim]);
            break;
          case SOLUTION_TIME_N:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->Set_Solution_time_n(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            break;
          case SOLUTION_TIME_N1:
            for (iVar = 0; iVar < nVar; iVar++)
              base_nodes->Set_Solution_time_n1(iPoint, iVar, bufDRecv[buf_offset+iVar]);
            break;
          default:
            SU2_MPI::Error("Unrecognized quantity for point-to-point MPI comms.",
                           CURRENT_FUNCTION);
            break;
        }
      }
      END_SU2_OMP_FOR
    }

    /*--- Verify that all non-blocking point-to-point sends have finished.
     Note that this should be satisfied, as we have received all of the
     data in the loop above at this point. ---*/

#ifdef HAVE_MPI
    SU2_OMP_SAFE_GLOBAL_ACCESS(SU2_MPI::Waitall(geometry->nP2PSend, geometry->req_P2PSend, MPI_STATUS_IGNORE);)
#endif
  }

}

void CSolver::ResetCFLAdapt() {
  NonLinRes_Series.clear();
  Old_Func = 0;
  New_Func = 0;
  NonLinRes_Counter = 0;
}


void CSolver::AdaptCFLNumber(CGeometry **geometry,
                             CSolver   ***solver_container,
                             CConfig   *config) {

  /* Adapt the CFL number on all multigrid levels using an
   exponential progression with under-relaxation approach. */

  vector<su2double> MGFactor(config->GetnMGLevels()+1,1.0);
  const su2double CFLFactorDecrease = config->GetCFL_AdaptParam(0);
  const su2double CFLFactorIncrease = config->GetCFL_AdaptParam(1);
  const su2double CFLMin            = config->GetCFL_AdaptParam(2);
  const su2double CFLMax            = config->GetCFL_AdaptParam(3);
  const su2double acceptableLinTol  = config->GetCFL_AdaptParam(4);
  const bool fullComms              = (config->GetComm_Level() == COMM_FULL);

  /* Number of iterations considered to check for stagnation. */
  const auto Res_Count = min(100ul, config->GetnInner_Iter()-1);

  static bool reduceCFL, resetCFL, canIncrease;

  for (unsigned short iMesh = 0; iMesh <= config->GetnMGLevels(); iMesh++) {

    /* Store the mean flow, and turbulence solvers more clearly. */

    CSolver *solverFlow = solver_container[iMesh][FLOW_SOL];
    CSolver *solverTurb = solver_container[iMesh][TURB_SOL];

    /* Compute the reduction factor for CFLs on the coarse levels. */

    if (iMesh == MESH_0) {
      MGFactor[iMesh] = 1.0;
    } else {
      const su2double CFLRatio = config->GetCFL(iMesh)/config->GetCFL(iMesh-1);
      MGFactor[iMesh] = MGFactor[iMesh-1]*CFLRatio;
    }

    /* Check whether we achieved the requested reduction in the linear
     solver residual within the specified number of linear iterations. */

    su2double linResTurb = 0.0;
    if ((iMesh == MESH_0) && solverTurb) linResTurb = solverTurb->GetResLinSolver();

    /* Max linear residual between flow and turbulence. */
    const su2double linRes = max(solverFlow->GetResLinSolver(), linResTurb);

    /* Tolerance limited to an acceptable value. */
    const su2double linTol = max(acceptableLinTol, config->GetLinear_Solver_Error());

    /* Check that we are meeting our nonlinear residual reduction target
     over time so that we do not get stuck in limit cycles, this is done
     on the fine grid and applied to all others. */

    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
    { /* Only the master thread updates the shared variables. */

    /* Check if we should decrease or if we can increase, the 20% is to avoid flip-flopping. */
    resetCFL = linRes > 0.99;
    reduceCFL = linRes > 1.2*linTol;
    canIncrease = linRes < linTol;

    if ((iMesh == MESH_0) && (Res_Count > 0)) {
      Old_Func = New_Func;
      if (NonLinRes_Series.empty()) NonLinRes_Series.resize(Res_Count,0.0);

      /* Sum the RMS residuals for all equations. */

      New_Func = 0.0;
      for (unsigned short iVar = 0; iVar < solverFlow->GetnVar(); iVar++) {
        New_Func += log10(solverFlow->GetRes_RMS(iVar));
      }
      if ((iMesh == MESH_0) && solverTurb) {
        for (unsigned short iVar = 0; iVar < solverTurb->GetnVar(); iVar++) {
          New_Func += log10(solverTurb->GetRes_RMS(iVar));
        }
      }

      /* Compute the difference in the nonlinear residuals between the
       current and previous iterations, taking care with very low initial
       residuals (due to initialization). */

      if ((config->GetInnerIter() == 1) && (New_Func - Old_Func > 10)) {
        Old_Func = New_Func;
      }
      NonLinRes_Series[NonLinRes_Counter] = New_Func - Old_Func;

      /* Increment the counter, if we hit the max size, then start over. */

      NonLinRes_Counter++;
      if (NonLinRes_Counter == Res_Count) NonLinRes_Counter = 0;

      /* Detect flip-flop convergence to reduce CFL and large increases
       to reset to minimum value, in that case clear the history. */

      if (config->GetInnerIter() >= Res_Count) {
        unsigned long signChanges = 0;
        su2double totalChange = 0.0;
        auto prev = NonLinRes_Series.front();
        for (auto val : NonLinRes_Series) {
          totalChange += val;
          signChanges += (prev > 0) ^ (val > 0);
          prev = val;
        }
        reduceCFL |= (signChanges > Res_Count/4) && (totalChange > -0.5);

        if (totalChange > 2.0) { // orders of magnitude
          resetCFL = true;
          NonLinRes_Counter = 0;
          for (auto& val : NonLinRes_Series) val = 0.0;
        }
      }
    }
    } /* End safe global access, now all threads update the CFL number. */
    END_SU2_OMP_SAFE_GLOBAL_ACCESS

    /* Loop over all points on this grid and apply CFL adaption. */

    su2double myCFLMin = 1e30, myCFLMax = 0.0, myCFLSum = 0.0;

    SU2_OMP_MASTER
    if ((iMesh == MESH_0) && fullComms) {
      Min_CFL_Local = 1e30;
      Max_CFL_Local = 0.0;
      Avg_CFL_Local = 0.0;
    }
    END_SU2_OMP_MASTER

    SU2_OMP_FOR_STAT(roundUpDiv(geometry[iMesh]->GetnPointDomain(),omp_get_max_threads()))
    for (unsigned long iPoint = 0; iPoint < geometry[iMesh]->GetnPointDomain(); iPoint++) {

      /* Get the current local flow CFL number at this point. */

      su2double CFL = solverFlow->GetNodes()->GetLocalCFL(iPoint);

      /* Get the current under-relaxation parameters that were computed
       during the previous nonlinear update. If we have a turbulence model,
       take the minimum under-relaxation parameter between the mean flow
       and turbulence systems. */

      su2double underRelaxationFlow = solverFlow->GetNodes()->GetUnderRelaxation(iPoint);
      su2double underRelaxationTurb = 1.0;
      if ((iMesh == MESH_0) && solverTurb)
        underRelaxationTurb = solverTurb->GetNodes()->GetUnderRelaxation(iPoint);
      const su2double underRelaxation = min(underRelaxationFlow,underRelaxationTurb);

      /* If we apply a small under-relaxation parameter for stability,
       then we should reduce the CFL before the next iteration. If we
       are able to add the entire nonlinear update (under-relaxation = 1)
       then we schedule an increase the CFL number for the next iteration. */

      su2double CFLFactor = 1.0;
      if (underRelaxation < 0.1 || reduceCFL) {
        CFLFactor = CFLFactorDecrease;
      } else if ((underRelaxation >= 0.1 && underRelaxation < 1.0) || !canIncrease) {
        CFLFactor = 1.0;
      } else {
        CFLFactor = CFLFactorIncrease;
      }

      /* Check if we are hitting the min or max and adjust. */

      if (CFL*CFLFactor <= CFLMin) {
        CFL       = CFLMin;
        CFLFactor = MGFactor[iMesh];
      } else if (CFL*CFLFactor >= CFLMax) {
        CFL       = CFLMax;
        CFLFactor = MGFactor[iMesh];
      }

      /* If we detect a stalled nonlinear residual, then force the CFL
       for all points to the minimum temporarily to restart the ramp. */

      if (resetCFL) {
        CFL       = CFLMin;
        CFLFactor = MGFactor[iMesh];
      }

      /* Apply the adjustment to the CFL and store local values. */

      CFL *= CFLFactor;
      solverFlow->GetNodes()->SetLocalCFL(iPoint, CFL);
      if ((iMesh == MESH_0) && solverTurb) {
        solverTurb->GetNodes()->SetLocalCFL(iPoint, CFL);
      }

      /* Store min and max CFL for reporting on the fine grid. */

      if ((iMesh == MESH_0) && fullComms) {
        myCFLMin = min(CFL,myCFLMin);
        myCFLMax = max(CFL,myCFLMax);
        myCFLSum += CFL;
      }

    }
    END_SU2_OMP_FOR

    /* Reduce the min/max/avg local CFL numbers. */

    if ((iMesh == MESH_0) && fullComms) {
      SU2_OMP_CRITICAL
      { /* OpenMP reduction. */
        Min_CFL_Local = min(Min_CFL_Local,myCFLMin);
        Max_CFL_Local = max(Max_CFL_Local,myCFLMax);
        Avg_CFL_Local += myCFLSum;
      }
      END_SU2_OMP_CRITICAL

      BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
      { /* MPI reduction. */
        myCFLMin = Min_CFL_Local; myCFLMax = Max_CFL_Local; myCFLSum = Avg_CFL_Local;
        SU2_MPI::Allreduce(&myCFLMin, &Min_CFL_Local, 1, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
        SU2_MPI::Allreduce(&myCFLMax, &Max_CFL_Local, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
        SU2_MPI::Allreduce(&myCFLSum, &Avg_CFL_Local, 1, MPI_DOUBLE, MPI_SUM, SU2_MPI::GetComm());
        Avg_CFL_Local /= su2double(geometry[iMesh]->GetGlobal_nPointDomain());
      }
      END_SU2_OMP_SAFE_GLOBAL_ACCESS
    }

  }

}

void CSolver::SetResidual_RMS(const CGeometry *geometry, const CConfig *config) {

  if (geometry->GetMGLevel() != MESH_0) return;

  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {

  /*--- Set the L2 Norm residual in all the processors. ---*/

  vector<su2double> rbuf_res(nVar);
  unsigned long Global_nPointDomain = 0;

  if (config->GetComm_Level() == COMM_FULL) {

    SU2_MPI::Allreduce(Residual_RMS.data(), rbuf_res.data(), nVar, MPI_DOUBLE, MPI_SUM, SU2_MPI::GetComm());
    Global_nPointDomain = geometry->GetGlobal_nPointDomain();
  }
  else {
    /*--- Reduced MPI comms have been requested. Use a local residual only. ---*/

    for (unsigned short iVar = 0; iVar < nVar; iVar++) rbuf_res[iVar] = Residual_RMS[iVar];
    Global_nPointDomain = geometry->GetnPointDomain();
  }

  for (unsigned short iVar = 0; iVar < nVar; iVar++) {

    if (std::isnan(SU2_TYPE::GetValue(rbuf_res[iVar]))) {
      SU2_MPI::Error("SU2 has diverged (NaN detected).", CURRENT_FUNCTION);
    }

    Residual_RMS[iVar] = max(EPS*EPS, sqrt(rbuf_res[iVar]/Global_nPointDomain));

    if (log10(GetRes_RMS(iVar)) > 20.0) {
      SU2_MPI::Error("SU2 has diverged (Residual > 10^20 detected).", CURRENT_FUNCTION);
    }
  }

  /*--- Set the Maximum residual in all the processors. ---*/

  if (config->GetComm_Level() == COMM_FULL) {

    const unsigned long nProcessor = size;

    su2activematrix rbuf_residual(nProcessor,nVar);
    su2matrix<unsigned long> rbuf_point(nProcessor,nVar);
    su2activematrix rbuf_coord(nProcessor*nVar, nDim);

    SU2_MPI::Allgather(Residual_Max.data(), nVar, MPI_DOUBLE, rbuf_residual.data(), nVar, MPI_DOUBLE, SU2_MPI::GetComm());
    SU2_MPI::Allgather(Point_Max.data(), nVar, MPI_UNSIGNED_LONG, rbuf_point.data(), nVar, MPI_UNSIGNED_LONG, SU2_MPI::GetComm());
    SU2_MPI::Allgather(Point_Max_Coord.data(), nVar*nDim, MPI_DOUBLE, rbuf_coord.data(), nVar*nDim, MPI_DOUBLE, SU2_MPI::GetComm());

    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      for (auto iProcessor = 0ul; iProcessor < nProcessor; iProcessor++) {
        AddRes_Max(iVar, rbuf_residual(iProcessor,iVar), rbuf_point(iProcessor,iVar), rbuf_coord[iProcessor*nVar+iVar]);
      }
    }
  }

  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS
}

void CSolver::SetResidual_BGS(const CGeometry *geometry, const CConfig *config) {

  if (geometry->GetMGLevel() != MESH_0) return;

  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {

  /*--- Set the L2 Norm residual in all the processors. ---*/

  vector<su2double> rbuf_res(nVar);

  SU2_MPI::Allreduce(Residual_BGS.data(), rbuf_res.data(), nVar, MPI_DOUBLE, MPI_SUM, SU2_MPI::GetComm());
  const auto Global_nPointDomain = geometry->GetGlobal_nPointDomain();

  for (unsigned short iVar = 0; iVar < nVar; iVar++) {
    Residual_BGS[iVar] = max(EPS*EPS, sqrt(rbuf_res[iVar]/Global_nPointDomain));
  }

  if (config->GetComm_Level() == COMM_FULL) {

    /*--- Set the Maximum residual in all the processors. ---*/

    const unsigned long nProcessor = size;

    su2activematrix rbuf_residual(nProcessor,nVar);
    su2matrix<unsigned long> rbuf_point(nProcessor,nVar);
    su2activematrix rbuf_coord(nProcessor*nVar, nDim);

    SU2_MPI::Allgather(Residual_Max_BGS.data(), nVar, MPI_DOUBLE, rbuf_residual.data(), nVar, MPI_DOUBLE, SU2_MPI::GetComm());
    SU2_MPI::Allgather(Point_Max_BGS.data(), nVar, MPI_UNSIGNED_LONG, rbuf_point.data(), nVar, MPI_UNSIGNED_LONG, SU2_MPI::GetComm());
    SU2_MPI::Allgather(Point_Max_Coord_BGS.data(), nVar*nDim, MPI_DOUBLE, rbuf_coord.data(), nVar*nDim, MPI_DOUBLE, SU2_MPI::GetComm());

    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      for (auto iProcessor = 0ul; iProcessor < nProcessor; iProcessor++) {
        AddRes_Max_BGS(iVar, rbuf_residual(iProcessor,iVar), rbuf_point(iProcessor,iVar), rbuf_coord[iProcessor*nVar+iVar]);
      }
    }
  }

  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS
}

void CSolver::SetRotatingFrame_GCL(CGeometry *geometry, const CConfig *config) {

  /*--- Loop interior points ---*/

  SU2_OMP_FOR_STAT(roundUpDiv(nPointDomain,2*omp_get_max_threads()))
  for (auto iPoint = 0ul; iPoint < nPointDomain; ++iPoint) {

    const su2double* GridVel_i = geometry->nodes->GetGridVel(iPoint);
    const su2double* Solution_i = base_nodes->GetSolution(iPoint);

    for (auto iNeigh = 0u; iNeigh < geometry->nodes->GetnPoint(iPoint); iNeigh++) {

      const auto iEdge = geometry->nodes->GetEdge(iPoint, iNeigh);
      const su2double* Normal = geometry->edges->GetNormal(iEdge);

      const auto jPoint = geometry->nodes->GetPoint(iPoint, iNeigh);
      const su2double* GridVel_j = geometry->nodes->GetGridVel(jPoint);

      /*--- Determine whether to consider the normal outward or inward. ---*/
      su2double dir = (iPoint < jPoint)? 0.5 : -0.5;

      su2double Flux = 0.0;
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Flux += dir*(GridVel_i[iDim]+GridVel_j[iDim])*Normal[iDim];

      for (auto iVar = 0u; iVar < nVar; iVar++)
        LinSysRes(iPoint,iVar) += Flux * Solution_i[iVar];
    }
  }
  END_SU2_OMP_FOR

  /*--- Loop boundary edges ---*/

  for (auto iMarker = 0u; iMarker < geometry->GetnMarker(); iMarker++) {
    if ((config->GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config->GetMarker_All_KindBC(iMarker) != NEARFIELD_BOUNDARY) &&
        (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (auto iVertex = 0u; iVertex < geometry->GetnVertex(iMarker); iVertex++) {

        const auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        /*--- Grid Velocity at each edge point ---*/

        const su2double* GridVel = geometry->nodes->GetGridVel(iPoint);

        /*--- Summed normal components ---*/

        const su2double* Normal = geometry->vertex[iMarker][iVertex]->GetNormal();

        su2double Flux = GeometryToolbox::DotProduct(nDim, Normal, GridVel);

        for (auto iVar = 0u; iVar < nVar; iVar++)
          LinSysRes(iPoint,iVar) -= Flux * base_nodes->GetSolution(iPoint,iVar);
      }
      END_SU2_OMP_FOR
    }
  }

}

void CSolver::SetAuxVar_Gradient_GG(CGeometry *geometry, const CConfig *config) {

  const auto& solution = base_nodes->GetAuxVar();
  auto& gradient = base_nodes->GetAuxVarGradient();

  computeGradientsGreenGauss(this, AUXVAR_GRADIENT, PERIODIC_NONE, *geometry,
                             *config, solution, 0, base_nodes->GetnAuxVar(), gradient);
}

void CSolver::SetAuxVar_Gradient_LS(CGeometry *geometry, const CConfig *config) {

  bool weighted = true;
  const auto& solution = base_nodes->GetAuxVar();
  auto& gradient = base_nodes->GetAuxVarGradient();
  auto& rmatrix  = base_nodes->GetRmatrix();

  computeGradientsLeastSquares(this, AUXVAR_GRADIENT, PERIODIC_NONE, *geometry, *config,
                               weighted, solution, 0, base_nodes->GetnAuxVar(), gradient, rmatrix);
}

void CSolver::SetSolution_Gradient_GG(CGeometry *geometry, const CConfig *config, bool reconstruction) {

  const auto& solution = base_nodes->GetSolution();
  auto& gradient = reconstruction? base_nodes->GetGradient_Reconstruction() : base_nodes->GetGradient();
  const auto comm = reconstruction? SOLUTION_GRAD_REC : SOLUTION_GRADIENT;
  const auto commPer = reconstruction? PERIODIC_SOL_GG_R : PERIODIC_SOL_GG;

  computeGradientsGreenGauss(this, comm, commPer, *geometry, *config, solution, 0, nVar, gradient);
}

void CSolver::SetSolution_Gradient_LS(CGeometry *geometry, const CConfig *config, bool reconstruction) {

  /*--- Set a flag for unweighted or weighted least-squares. ---*/
  bool weighted;
  PERIODIC_QUANTITIES commPer;

  if (reconstruction) {
    weighted = (config->GetKind_Gradient_Method_Recon() == WEIGHTED_LEAST_SQUARES);
    commPer = weighted? PERIODIC_SOL_LS_R : PERIODIC_SOL_ULS_R;
  }
  else {
    weighted = (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES);
    commPer = weighted? PERIODIC_SOL_LS : PERIODIC_SOL_ULS;
  }

  const auto& solution = base_nodes->GetSolution();
  auto& rmatrix = base_nodes->GetRmatrix();
  auto& gradient = reconstruction? base_nodes->GetGradient_Reconstruction() : base_nodes->GetGradient();
  const auto comm = reconstruction? SOLUTION_GRAD_REC : SOLUTION_GRADIENT;

  computeGradientsLeastSquares(this, comm, commPer, *geometry, *config, weighted, solution, 0, nVar, gradient, rmatrix);
}

void CSolver::SetUndivided_Laplacian(CGeometry *geometry, const CConfig *config) {

  /*--- Loop domain points. ---*/

  SU2_OMP_FOR_DYN(256)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; ++iPoint) {

    const bool boundary_i = geometry->nodes->GetPhysicalBoundary(iPoint);

    /*--- Initialize. ---*/
    for (unsigned short iVar = 0; iVar < nVar; iVar++)
      base_nodes->SetUnd_Lapl(iPoint, iVar, 0.0);

    /*--- Loop over the neighbors of point i. ---*/
    for (auto jPoint : geometry->nodes->GetPoints(iPoint)) {

      bool boundary_j = geometry->nodes->GetPhysicalBoundary(jPoint);

      /*--- If iPoint is boundary it only takes contributions from other boundary points. ---*/
      if (boundary_i && !boundary_j) continue;

      /*--- Add solution differences, with correction for compressible flows which use the enthalpy. ---*/

      for (unsigned short iVar = 0; iVar < nVar; iVar++) {
        su2double delta = base_nodes->GetSolution(jPoint,iVar)-base_nodes->GetSolution(iPoint,iVar);
        base_nodes->AddUnd_Lapl(iPoint, iVar, delta);
      }
    }
  }
  END_SU2_OMP_FOR

  /*--- Correct the Laplacian across any periodic boundaries. ---*/

  for (unsigned short iPeriodic = 1; iPeriodic <= config->GetnMarker_Periodic()/2; iPeriodic++) {
    InitiatePeriodicComms(geometry, config, iPeriodic, PERIODIC_LAPLACIAN);
    CompletePeriodicComms(geometry, config, iPeriodic, PERIODIC_LAPLACIAN);
  }

  /*--- MPI parallelization ---*/

  InitiateComms(geometry, config, UNDIVIDED_LAPLACIAN);
  CompleteComms(geometry, config, UNDIVIDED_LAPLACIAN);

}

void CSolver::Add_External_To_Solution() {
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {
    base_nodes->AddSolution(iPoint, base_nodes->Get_External(iPoint));
  }

  base_nodes->Add_ExternalExtra_To_SolutionExtra();
}

void CSolver::Add_Solution_To_External() {
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {
    base_nodes->Add_External(iPoint, base_nodes->GetSolution(iPoint));
  }

  base_nodes->Set_ExternalExtra_To_SolutionExtra();
}

void CSolver::Update_Cross_Term(CConfig *config, su2passivematrix &cross_term) {

  /*--- This method is for discrete adjoint solvers and it is used in multi-physics
   *    contexts, "cross_term" is the old value, the new one is in "Solution".
   *    We update "cross_term" and the sum of all cross terms (in "External")
   *    with a fraction of the difference between new and old.
   *    When "alpha" is 1, i.e. no relaxation, we effectively subtract the old
   *    value and add the new one to the total ("External"). ---*/

  vector<su2double> solution(nVar);
  passivedouble alpha = SU2_TYPE::GetValue(config->GetAitkenStatRelax());

  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {
    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      passivedouble
      new_val = SU2_TYPE::GetValue(base_nodes->GetSolution(iPoint,iVar)),
      delta = alpha * (new_val - cross_term(iPoint,iVar));
      /*--- Update cross term. ---*/
      cross_term(iPoint,iVar) += delta;
      solution[iVar] = delta;
    }
    /*--- Update the sum of all cross-terms. ---*/
    base_nodes->Add_External(iPoint, solution.data());
  }
}

void CSolver::SetGridVel_Gradient(CGeometry *geometry, const CConfig *config) {

  /// TODO: No comms needed for this gradient? The Rmatrix should be allocated somewhere.

  const auto& gridVel = geometry->nodes->GetGridVel();
  auto& gridVelGrad = geometry->nodes->GetGridVel_Grad();
  auto rmatrix = CVectorOfMatrix(nPoint,nDim,nDim);

  computeGradientsLeastSquares(nullptr, GRID_VELOCITY, PERIODIC_NONE, *geometry, *config,
                               true, gridVel, 0, nDim, gridVelGrad, rmatrix);
}

void CSolver::SetSolution_Limiter(CGeometry *geometry, const CConfig *config) {

  const auto kindLimiter = config->GetKind_SlopeLimit();
  const auto& solution = base_nodes->GetSolution();
  const auto& gradient = base_nodes->GetGradient_Reconstruction();
  auto& solMin = base_nodes->GetSolution_Min();
  auto& solMax = base_nodes->GetSolution_Max();
  auto& limiter = base_nodes->GetLimiter();

  computeLimiters(kindLimiter, this, SOLUTION_LIMITER, PERIODIC_LIM_SOL_1, PERIODIC_LIM_SOL_2,
                  *geometry, *config, 0, nVar, solution, gradient, solMin, solMax, limiter);
}

void CSolver::Gauss_Elimination(su2double** A, su2double* rhs, unsigned short nVar) {

  short iVar, jVar, kVar;
  su2double weight, aux;

  if (nVar == 1)
    rhs[0] /= A[0][0];
  else {

    /*--- Transform system in Upper Matrix ---*/

    for (iVar = 1; iVar < (short)nVar; iVar++) {
      for (jVar = 0; jVar < iVar; jVar++) {
        weight = A[iVar][jVar]/A[jVar][jVar];
        for (kVar = jVar; kVar < (short)nVar; kVar++)
          A[iVar][kVar] -= weight*A[jVar][kVar];
        rhs[iVar] -= weight*rhs[jVar];
      }
    }

    /*--- Backwards substitution ---*/

    rhs[nVar-1] = rhs[nVar-1]/A[nVar-1][nVar-1];
    for (iVar = (short)nVar-2; iVar >= 0; iVar--) {
      aux = 0;
      for (jVar = iVar+1; jVar < (short)nVar; jVar++)
        aux += A[iVar][jVar]*rhs[jVar];
      rhs[iVar] = (rhs[iVar]-aux)/A[iVar][iVar];
      if (iVar == 0) break;
    }
  }

}

void CSolver::Gauss_Elimination(vector<vector<su2double>>& A,vector<su2double>& sol) {
    int n = A.size();

    for (int i=0; i<n; i++) {
        // Search for maximum in this column
        su2double maxEl = abs(A[i][i]);
        int maxRow = i;
        for (int k=i+1; k<n; k++) {
            if (abs(A[k][i]) > maxEl) {
                maxEl = abs(A[k][i]);
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        for (int k=i; k<n+1;k++) {
            su2double tmp = A[maxRow][k];
            A[maxRow][k] = A[i][k];
            A[i][k] = tmp;
        }

        // Make all rows below this one 0 in current column
        for (int k=i+1; k<n; k++) {
            su2double c = -A[k][i]/A[i][i];
            for (int j=i; j<n+1; j++) {
                if (i==j) {
                    A[k][j] = 0;
                } else {
                    A[k][j] += c * A[i][j];
                }
            }
        }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    //vector<su2double> x(n);
    for (int i=n-1; i>=0; i--) {
        sol[i] = A[i][n]/A[i][i];
        for (int k=i-1;k>=0; k--) {
            A[k][n] -= A[k][i] * sol[i];
        }
    }
    
}

void CSolver::Inverse_matrix2D(vector<vector<su2double>> &Phi, vector<vector<su2double>> &Phi_inv){

  vector<vector<su2double>> Phi_sol(2,vector<su2double>(2,0.0));
  vector<vector<su2double>> Phi_co(2,vector<su2double>(2,0.0));
  vector<vector<su2double>> subVect(1,vector<su2double>(1,0.0));
//
  su2double Det_Phi= Phi[0][0] * Phi[1][1] - Phi[0][1] * Phi[1][0], Det_subVect;

  su2double d = 1.0/Det_Phi; 
   
  for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {

            int p = 0;
            for(int l = 0; l < 2; l++) {
                if(l == i) continue;
             
                int q = 0;

                for(int m = 0; m < 2; m++) {
                    if(m == j) continue;

                    subVect[p][q] = Phi[l][m];
                    q++;
                }
                p++;
            }
	    Det_subVect = subVect[0][0];

            Phi_co[i][j] = pow(-1, i + j) * Det_subVect;
        }  
  }
  
  for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            Phi_sol[j][i] = Phi_co[i][j];
        }  
  }

  for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            Phi_inv[i][j] = Phi_sol[i][j] * d; 
        }  
  }

//cout << "Phi" << endl;
//    for(int j=0;j<2;j++) { 
//           cout << Phi[j][0] << "  "<< Phi[j][1] << "\n";
//  }
//    cout << endl;
   
//  cout << "Phi_inverse" << endl;
//    for(int j=0;j<2;j++) { 
//           cout << Phi_inv[j][0] << "  "<< Phi_inv[j][1] << "\n";
//  }
//    cout << endl;
  
//
}

void CSolver::Aeroelastic(CSurfaceMovement *surface_movement, CGeometry *geometry, CConfig *config, unsigned long TimeIter) {

  /*--- Variables used for Aeroelastic case ---*/

  su2double Cl, Cd, Cn, Ct, Cm, Cn_rot;
  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  vector<su2double> structural_solution(4,0.0); //contains solution(displacements and rates) of typical section wing model.

  unsigned short iMarker, iMarker_Monitoring, Monitoring;
  string Marker_Tag, Monitoring_Tag;

  /*--- Loop over markers and find the ones being monitored. ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    Monitoring = config->GetMarker_All_Monitoring(iMarker);
    if (Monitoring == YES) {

      /*--- Find the particular marker being monitored and get the forces acting on it. ---*/

      for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
        Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
        Marker_Tag = config->GetMarker_All_TagBound(iMarker);
        if (Marker_Tag == Monitoring_Tag) {

          Cl = GetSurface_CL(iMarker_Monitoring);
          Cd = GetSurface_CD(iMarker_Monitoring);

          /*--- For typical section wing model want the force normal to the airfoil (in the direction of the spring) ---*/
          Cn = Cl*cos(Alpha) + Cd*sin(Alpha);
          Ct = -Cl*sin(Alpha) + Cd*cos(Alpha);

          Cm = GetSurface_CMz(iMarker_Monitoring);

          /*--- Calculate forces for the Typical Section Wing Model taking into account rotation ---*/

          /*--- Note that the calculation of the forces and the subsequent displacements ...
           is only correct for the airfoil that starts at the 0 degree position ---*/

          if (config->GetKind_GridMovement() == AEROELASTIC_RIGID_MOTION) {
            su2double Omega, dt, psi;
            dt = config->GetDelta_UnstTimeND();
            Omega  = (config->GetRotation_Rate(2)/config->GetOmega_Ref());
            psi = Omega*(dt*TimeIter);

            /*--- Correct for the airfoil starting position (This is hardcoded in here) ---*/
            if (Monitoring_Tag == "Airfoil1") {
              psi = psi + 0.0;
            }
            else if (Monitoring_Tag == "Airfoil2") {
              psi = psi + 2.0/3.0*PI_NUMBER;
            }
            else if (Monitoring_Tag == "Airfoil3") {
              psi = psi + 4.0/3.0*PI_NUMBER;
            }
            else
              cout << "WARNING: There is a marker that we are monitoring that doesn't match the values hardcoded above!" << endl;

            cout << Monitoring_Tag << " position " << psi*180.0/PI_NUMBER << " degrees. " << endl;

            Cn_rot = Cn*cos(psi) - Ct*sin(psi); //Note the signs are different for accounting for the AOA.
            Cn = Cn_rot;
          }

          /*--- Solve the aeroelastic equations for the particular marker(surface) ---*/

          SolveTypicalSectionWingModel(geometry, Cn, Cm, config, iMarker_Monitoring, structural_solution);

          break;
        }
      }

      /*--- Compute the new surface node locations ---*/
      surface_movement->AeroelasticDeform(geometry, config, TimeIter, iMarker, iMarker_Monitoring, structural_solution);

    }

  }

}


void CSolver::Aeroelastic_HB(su2double**& aero_solutions, CGeometry *geometry, CConfig *config, CSolver**** flow_solution, int Instances) {

  /*--- Variables used for Aeroelastic case ---*/
 
  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  const su2double DEG2RAD = PI_NUMBER/180.0;
  su2double time_new;
  vector<vector<su2double>> aero_hb_sol(Instances, vector<su2double>(4,0.0));
  vector<su2double> Cl(Instances,0.0), Cn(Instances,0.0); //contains solution(displacements and rates) of typical section wing model.
  vector<su2double> Cd(Instances,0.0), Ct(Instances,0.0); //contains solution(displacements and rates) of typical section wing model.
  vector<su2double> Cm(Instances,0.0); //
  vector<su2double> Ampl(3,0.0), Omega(3, 0.0);

  unsigned short iMarker, iMarker_Monitoring, Monitoring;
  string Marker_Tag, Monitoring_Tag;

  if (rank == MASTER_NODE) cout << endl << "-------------------- Aeroelastic HB Displ. Computation --------------------" << endl;


    for (int iInst = 0; iInst < Instances; iInst++){
 
      Cl[iInst] = flow_solution[iInst][MESH_0][FLOW_SOL]->GetTotal_CL();
      Cd[iInst] = flow_solution[iInst][MESH_0][FLOW_SOL]->GetTotal_CD();
      Cm[iInst] = flow_solution[iInst][MESH_0][FLOW_SOL]->GetTotal_CMz(); 

    }
   
    for(int iInst=0;iInst<Instances;iInst++) {

          Cn[iInst] =  Cl[iInst]*cos(Alpha) + Cd[iInst]*sin(Alpha);
          Ct[iInst] = -Cl[iInst]*sin(Alpha) + Cd[iInst]*cos(Alpha);

    } 
       
        su2double period = config->GetHarmonicBalance_Period();
    	period /= config->GetTime_Ref();
        su2double TimeInstances = config->GetnTimeInstances();
        su2double deltaT = period/TimeInstances;
 
  /*--- Loop over markers and find the ones being monitored. ---*/
 

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    Monitoring = config->GetMarker_All_Monitoring(iMarker);
    if (Monitoring == YES) {

      /*--- Find the particular marker being monitored and get the forces acting on it. ---*/

      for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
        Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
        Marker_Tag = config->GetMarker_All_TagBound(iMarker);
        if (Marker_Tag == Monitoring_Tag) {


///// for (int iInst=0; iInst<Instances; iInst++) {

/////         time_new = iInst*deltaT;

/////         for (int iDim = 0; iDim < 3; iDim++){
/////           Omega[iDim] = config->GetPlunging_Omega(iDim)/config->GetOmega_Ref();
/////           Ampl[iDim]  = config->GetPlunging_Ampl(iDim);             
/////         }

/////         aero_hb_sol[iInst][0]   =  Ampl[1]*sin(Omega[1]*time_new);
/////         aero_hb_sol[iInst][2]   =  Omega[1]*Ampl[1]*cos(Omega[1]*time_new);

/////         for (int iDim = 0; iDim < 3; iDim++){
/////   	Omega[iDim] = config->GetPitching_Omega(iDim)/config->GetOmega_Ref();
/////           Ampl[iDim]  = config->GetPitching_Ampl(iDim)*DEG2RAD;             

/////        }

/////         aero_hb_sol[iInst][1]   =  Ampl[2]*sin(Omega[2]*time_new);
/////         aero_hb_sol[iInst][3]   =  Omega[2]*Ampl[2]*cos(Omega[2]*time_new);

///// }

      SolveWing_HB_Unst2D(geometry, config, iMarker_Monitoring, aero_hb_sol, Cn, Ct, Cm, Instances);
      //
      //SolveWing_HB_Thomas_Flutter(geometry, config, iMarker_Monitoring, aero_hb_sol, Cn, Ct, Cm, Instances);
      //
      //SolveWing_HB_Thomas_Velocity(geometry, config, iMarker_Monitoring, aero_hb_sol, Cn, Ct, Cm, Instances);

      for (int iInst=0; iInst<Instances; iInst++) {

            aero_solutions[iInst][0] = aero_hb_sol[iInst][0];
            aero_solutions[iInst][1] = aero_hb_sol[iInst][1];
            aero_solutions[iInst][2] = aero_hb_sol[iInst][2];
            aero_solutions[iInst][3] = aero_hb_sol[iInst][3];
        
      }


   }

 
   
   }
        
   }
    
  }
}

void CSolver::AeroelasticWing(su2double* &structural_solution, su2double* gen_forces, CGeometry *geometry, CConfig *config, unsigned long TimeIter) {

  /*--- Variables used for Aeroelastic case ---*/

  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  su2double deltaT = config->GetDelta_UnstTimeND(), time_new, time_old, Cl, Cd;
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> OMEGA(4,0.0), MODE(4,0.0);
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);

  ///(modes,0.0); //contains solution(displacements and rates) of typical section wing model.
  ///(modes,0.0); //contains generalized forces

  unsigned short iMarker, iMarker_Monitoring, Monitoring;
  string Marker_Tag, Monitoring_Tag;

  /*--- Loop over markers and find the ones being monitored. ---*/

  if (harmonic_balance) {
    /*--- period of oscillation & time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period/TimeInstances;
  }

  if (TimeIter == 0) {

	  time_new = 0.0;
          time_old = 0.0; 
  }
  else {
	  time_new = TimeIter*deltaT;
	  time_old = (TimeIter-1)*deltaT;
  }
 	  
  if (harmonic_balance) time_old = 0.0;

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    Monitoring = config->GetMarker_All_Monitoring(iMarker);
    if (Monitoring == YES) {

      /*--- Find the particular marker being monitored and get the forces acting on it. ---*/

      for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
        Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
        Marker_Tag = config->GetMarker_All_TagBound(iMarker);
        if (Marker_Tag == Monitoring_Tag) {          

          /*--- Solve the aeroelastic equations for the particular marker(surface) ---*/

          Cl = GetSurface_CL(iMarker_Monitoring);
          Cd = GetSurface_CD(iMarker_Monitoring);

          if (rank == MASTER_NODE) cout << "SU2 Calc CL= " << Cl << " | CD= " << Cd << endl;
         
		if (!config->GetImposed_Modal_Move() && !harmonic_balance){

		        	
			SolveModalWing(geometry, config, iMarker_Monitoring, gen_forces, structural_solution);

		}
		else {
          
                	for (unsigned short iDim = 0; iDim < modes; iDim++){
        
				MODE[iDim]   = config->GetMode_Ampl(iDim);
        
				OMEGA[iDim]  = config->GetMode_Omega(iDim);

				structural_solution[iDim] = MODE[iDim]*(sin(OMEGA[iDim]*time_new) - sin(OMEGA[iDim]*time_old));	
		  	}
		

		}

          break;

        }
      }

      /*--- Compute the new surface node locations ---*/
      //surface_movement->AeroelasticDeformWing(geometry, config, TimeIter, iMarker, iMarker_Monitoring, structural_solution);

    }

  }

}

void CSolver::AeroelasticWing_HB(su2double** &structural_solution, su2double**& gen_forces, CGeometry *geometry, CConfig *config, int harmonics) {

  /*--- Variables used for Aeroelastic case ---*/

  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  su2double deltaT, time_new, time_old, Cl, Cd;
  unsigned short modes = config->GetNumber_Modes();
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);

  unsigned short iMarker, iMarker_Monitoring, Monitoring;
  string Marker_Tag, Monitoring_Tag;

  /*--- Loop over markers and find the ones being monitored. ---*/

  if (harmonic_balance) {
    /*--- period of oscillation & time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period/TimeInstances;
  }

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    Monitoring = config->GetMarker_All_Monitoring(iMarker);
    if (Monitoring == YES) {

      /*--- Find the particular marker being monitored and get the forces acting on it. ---*/

      for (iMarker_Monitoring = 0; iMarker_Monitoring < config->GetnMarker_Monitoring(); iMarker_Monitoring++) {
        Monitoring_Tag = config->GetMarker_Monitoring_TagBound(iMarker_Monitoring);
        Marker_Tag = config->GetMarker_All_TagBound(iMarker);
        if (Marker_Tag == Monitoring_Tag) {          

          /*--- Solve the aeroelastic equations for the particular marker(surface) ---*/
        	
		SolveModalWing_HB(geometry, config, iMarker_Monitoring, gen_forces, structural_solution, harmonics);

        }
      }

      /*--- Compute the new surface node locations ---*/
      //surface_movement->AeroelasticDeformWing(geometry, config, TimeIter, iMarker, iMarker_Monitoring, structural_solution);

    }

  }

}

void CSolver::SolveWing_HB_Unst2D(CGeometry *geometry, CConfig *config, unsigned short iMarker, vector<vector<su2double>>& displacements, vector<su2double> cl, vector<su2double> cd, vector<su2double> cm, int harmonics) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/

  /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double DEG2RAD = PI_NUMBER/180.0;
  su2double w_alpha = w_a;
  su2double vf       = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b        = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  unsigned long TimeIter = config->GetTimeIter();
  su2double dtau;

  cout << "xa= " << x_a << ", r2a= " << r_a*r_a << ", wh/wa= " << wr << ", Vf= " << vf << ", b= " << b << endl;

  //su2double dt      = config->GetDelta_UnstTimeND();
//  dt = dt*w_alpha; //Non-dimensionalize the structural time.
  //
  int dofs = 4, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(2,0.0);

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  /*--- Eigenvectors and Eigenvalues of the Generalized EigenValue Problem. ---*/
 vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));

 // Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  Kdofs[0][2] = -1.0;
  Kdofs[1][3] = -1.0;

  Kdofs[2][0] = AK[0][0];
  Kdofs[2][1] = AK[0][1];
  Kdofs[3][0] = AK[1][0];
  Kdofs[3][1] = AK[1][1];

  Kdofs[2][2] = AT[0][0];
  Kdofs[2][3] = AT[0][1];
  Kdofs[3][2] = AT[1][0];
  Kdofs[3][3] = AT[1][1];

  //
  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0)); 
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = (DD[j][k]*OmegaHB)/w_alpha;
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }


  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }

//  
  vector<vector<su2double>> AHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        AHB[i][j] += Qt[i][k] * tmp[k][j];
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }
  } 

  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = Kdofs[j][k];
	  }
	  }
	
  }

  /*--- Solving the Decoupled Aeroelastic Problem with second order time discretization Eq (9) ---*/

 /*--- Set up of variables used to solve the structural problem. ---*/
  
//  vector<vector<su2double> > A_inv(2,vector<su2double>(2,0.0));
//  su2double detA;
//  su2double s1, s2;
//  vector<su2double> rhs(2,0.0); //right hand side
 
    /*--- Forcing Term ---*/
  vector<su2double> Force(NT*dofs,0);
  vector<su2double> Force_old(NT*dofs,0); 
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);
  vector<su2double> eta_old(NT*dofs,0.0);
  vector<su2double> eta_new(NT*dofs,0.0);  
  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));

//
//     
//
  cout << "LIFT   /    MOMENT" << endl;
  for(int count=0;count<NT;count++){
  
	  cout << cl[count] << " " << cm[count] << endl;

  f[0] = cons*(-cl[count]);
  f[1] = cons*(-2*cm[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += M_inv[i][k]*f[k]; //PHI transpose
    }
  }

  Force[count*dofs+2] = f_tilde[0];
  Force[count*dofs+3] = f_tilde[1];

  }
	
//for(int i=0;i<NT*dofs;i++) {

//    eta_old[i] = config->GetHB_displacements(i);


//}

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
    q_dot[1][i] = config->GetHB_pitch_rate(i)/w_alpha;
    q_dot[0][i] = config->GetHB_plunge_rate(i)/b/w_alpha;
  }
  int kk = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta_old[i+0] = q[0][kk];
          eta_old[i+1] = q[1][kk];

          eta_old[i+2] = q_dot[0][kk];
          eta_old[i+3] = q_dot[1][kk];

          kk += 1;
  }

//for(int i=0;i<NT*dofs;i++) {

//    Force_old[i] = config->GetHB_forces(i);

//}

  vector<vector<su2double> > ASys(NT*dofs,vector<su2double>(NT*dofs+1,0.0));
  dtau = config->GetPseudoTimeStep(); 
 
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {

         ASys[i][j] = dtau*(AHB[i][j] + KHB[i][j]);
         if (i==j) ASys[i][i] +=  1.0;

      }      
  }

  for(int j=0;j<NT*dofs;j++) {

        //ASys[j][NT*dofs] = eta_old[j] +  0.5*dtau*(Force[j] + Force_old[j]);
        ASys[j][NT*dofs] = eta_old[j] +  dtau*Force[j];

  }

  Gauss_Elimination(ASys, eta);


    cout << "eta_old" << endl;
    for(int j=0;j<NT*dofs;j++) { 
             cout << eta_old[j] << "\n";
    }
    cout << endl;

    cout << "eta" << endl;
    for(int j=0;j<NT*dofs;j++) { 
             cout << eta[j] << "\n";
    }
    cout << endl;

//for(int i=0;i<NT*dofs;i++) {

//    config->SetHB_forces(Force[i], i);

//}
//for(int i=0;i<NT*dofs;i++) {

//    config->SetHB_displacements(eta[i], i);

//}

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

	  q[0][k] = eta[i+0];
	  q[1][k] = eta[i+1];

	  q_dot[0][k] = eta[i+2];
	  q_dot[1][k] = eta[i+3];

	  k += 1;
  } 
 

  //

    cout << "q" << endl;
      for(int j=0;j<NT;j++) { 
             cout << q[0][j] << "  "<< q[1][j] << "\n";
    }
      cout << endl;
 
      cout << "q DOT" << endl;
      for(int j=0;j<NT;j++) { 
             cout << q_dot[0][j] << "  "<< q_dot[1][j] << "\n";
    }
      cout << endl;


  // PHASE FIX
// 
  vector<su2double> hh(NT,0.0);
  vector<su2double> ah(NT,0.0);
  vector<su2double> hhd(NT,0.0);
  vector<su2double> ahd(NT,0.0);

  cout << "a_hat  |  h_hat" << endl;
  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       hh[i] += EE[i][j]*q[0][j];
       ah[i] += EE[i][j]*q[1][j];
  
       hhd[i] += EE[i][j]*q_dot[0][j];
       ahd[i] += EE[i][j]*q_dot[1][j];
       
  }
  cout << ah[i] << " | "  << hh[i] << "\n";

  }
  cout << endl;

////////    su2double U0  = hh[0], Cc1 = hh[1], Cs1 = hh[2], Cc2 = hh[3], Cs2 = hh[4];
            su2double U0  = ah[0], Cc1 = ah[1], Cs1 = ah[2]; ///// Cc2 = ah[3], Cs2 = ah[4];

           // su2double phase = atan2(Cs1,Cc1) - PI_NUMBER/2;
            su2double magn  = sqrt(Cc1*Cc1 + Cs1*Cs1);

            su2double ratio = config->GetPitching_Ampl(2)*DEG2RAD/magn;

	    cout << "Ratio= " << ratio << endl;
                
 //////  hh[0] = 0.0;
 //////  ah[0] = 0.0;
 //////  hhd[0] = 0.0;
 //////  ahd[0] = 0.0;
 
	 if (config->HB_Flutter()) { 

           if (ratio < 1.0) { ah[0] = ah[0]*ratio;
                             // hh[0] = hh[0]*ratio;
                              ahd[0] = ahd[0]*ratio;
                            //  hhd[0] = hhd[0]*ratio; 
                            }
         
     ////ah[0] = 0.0; 
     ////hh[0] = 0.0;
     ////ahd[0] = 0.0;
     ////hhd[0] = 0.0; 

         //ah[0] = 0.0;
         //ah[1] = ah[1]*ratio;
         //ah[2] = ah[2]*ratio;

         //if (ratio < 1.0) ahd[0] = ahd[0]*ratio;
         //ahd[0] = 0.0;
         //ahd[1] = ahd[1]*ratio;
         //ahd[2] = ahd[2]*ratio;

         for (int i=1;i<NT;i++) {
          
          ah[i]  = ah[i]*ratio;
          ahd[i] = ahd[i]*ratio;
       //   hh[i]  = hh[i]*ratio;
       //   hhd[i] = hhd[i]*ratio; 

         }

//////// if (ratio < 1.0) hh[0] = hh[0]*ratio;
//////// //hh[0] = 0.0;
//////// hh[1] = hh[1]*ratio;
//////// hh[2] = hh[2]*ratio;

         }

  cout << "a_hat  |  h_hat" << endl;
  for(int i=0;i<NT;i++) { 

      cout << ah[i] << " | "  << hh[i] << "\n";

  }
  cout << endl;


            U0  = ah[0], Cc1 = ah[1], Cs1 = ah[2];

            su2double phase = atan2(Cs1,Cc1) - PI_NUMBER/2;
    
            su2double dt_lag = phase/OmegaHB;
            cout << "phase= " << phase << endl;

            vector<su2double> time_hb(NT,0.0), time_hb2(NT,0.0);
            for(int i=0;i<NT;i++) {

                    time_hb[i] = i*Period/NT + dt_lag;
                    //time_hb[i] = i*Period/NT;
            }
      

         vector<su2double> h(NT,0.0);
         vector<su2double> alpha(NT,0.0);
         vector<su2double> h_dot(NT,0.0);
         vector<su2double> alpha_dot(NT,0.0);

            for(int i=0;i<NT;i++) {

                    h[i]     = hh[0];
                    alpha[i] = ah[0];  
                    h_dot[i] = hhd[0];
                    alpha_dot[i] = ahd[0];

                    for(int j=0;j<NH;j++) {

                    h[i]     = h[i] + hh[2*j+1]*cos((j+1)*OmegaHB*time_hb[i]) 
                	            + hh[2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);
                    alpha[i] = alpha[i] + ah[2*j+1]*cos((j+1)*OmegaHB*time_hb[i])
                		        + ah[2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);  
                    h_dot[i] = h_dot[i] + hhd[2*j+1]*cos((j+1)*OmegaHB*time_hb[i]) 
                	                + hhd[2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);
                    alpha_dot[i] = alpha_dot[i] + ahd[2*j+1]*cos((j+1)*OmegaHB*time_hb[i]) 
                	                        + ahd[2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);
            
                    }

            }


   /*--- Set the solution of the structural equations ---*/
//if (config->HB_LCO()) { 
//for (int i=0;i<NT;i++) {

//displacements[i][0] = b*q[0][i];
//displacements[i][1] = q[1][i];
//displacements[i][2] = w_alpha*b*q_dot[0][i];
//displacements[i][3] = w_alpha*q_dot[1][i];

//}
//}
//else if (config->HB_Flutter()) {
  for (int i=0;i<NT;i++) {

  displacements[i][0] = b*h[i];
  displacements[i][1] = alpha[i];
  displacements[i][2] = w_alpha*b*h_dot[i];
  displacements[i][3] = w_alpha*alpha_dot[i];

  }
//}

  for (int i=0;i<NT;i++){  
    config->SetHB_pitch(displacements[i][1], i);
    config->SetHB_plunge(displacements[i][0]/b, i);
    config->SetHB_pitch_rate(displacements[i][3], i);
    config->SetHB_plunge_rate(displacements[i][2], i);
  }


}

void CSolver::SolveModalWing_HB(CGeometry *geometry, CConfig *config, unsigned short iMarker, su2double**& gen_forces, su2double**& gen_displacements, int harmonics) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/

  /*--- Retrieve values from the config file ---*/
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> w_modes(modes,0.0); //contains solution(displacements and rates) of typical section wing model.

  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetRefWing_Length()/2.0; // root airfoil semichord
  su2double Vo      = config->GetConicalRefVol();
  su2double scale_param = 1/config->Get_Scaling_Parameter();

  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
 
  su2double DEG2RAD = PI_NUMBER/180.0;
  su2double w_alpha = w_a;

  cout << "Solving Wing (HB)" << endl;

  int dofs = 2*modes, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(modes,0.0);
  for (unsigned short i=0;i<modes; i++) {

          w_modes[i] = config->GetAero_Omega(i)/w_a;

  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  /*--- Eigenvectors and Eigenvalues of the Generalized EigenValue Problem. ---*/
  vector<vector<su2double>> AK(dofs, vector<su2double>(dofs,0.0));

  // Stiffness Matrix
  vector<vector<su2double> > K(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          K[i][i] = w_modes[i]*w_modes[i];

  }

   // Stiffness Matrix
  vector<vector<su2double> > T(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          T[i][i] = 2.0*xi[i]*w_modes[i];

  }

  for (unsigned short i=0;i<modes; i++) {

          AK[i][i+modes] = -1.0;

  }
  for (unsigned short i=0;i<modes; i++) {

          AK[i+modes][i] = K[i][i];

  }
  for (unsigned short i=0;i<modes; i++) {

          AK[i+modes][i+modes] = T[i][i];

  }
  //
  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0)); 
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k]*(OmegaHB/w_alpha);
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }


  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }

//  
  vector<vector<su2double>> AHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        AHB[i][j] += Qt[i][k] * tmp[k][j];
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }
  } 

  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = AK[j][k];
	  }
	  }
	
  }

  /*--- Solving the Decoupled Aeroelastic Problem with second order time discretization Eq (9) ---*/

 /*--- Set up of variables used to solve the structural problem. ---*/
  
    /*--- Forcing Term ---*/
  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf*b*b/2.0/Vo;
  vector<su2double> eta(NT*dofs,0.0);
  vector<su2double> eta_old(NT*dofs,0.0);
  vector<su2double> eta_new(NT*dofs,0.0);  
  vector<vector<su2double>> q(modes,vector<su2double>(NT,0.0));
  vector<vector<su2double>> q_dot(modes,vector<su2double>(NT,0.0));

//
  cout << "Calculating Force vector..." << endl; 
  for(int count=0;count<NT;count++){
  for (int i=0; i<modes; i++) {
   
    Force[count*dofs+modes+i] = (cons * scale_param * gen_forces[count][i]);
    
  }
  }

  cout << "Reading old disp. vector..." << endl;	
  for (int i=0;i<NT;i++){  
  for (int j=0;j<modes;j++) {
    q[j][i]     = config->GetHB_Modal_Displacement(i, j);
    q_dot[j][i] = config->GetHB_Modal_Velocities(i, j);
  }
  }

  int l = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {
  for(int j=0;j<modes;j++) {

	  eta_old[i+j] = q[j][l];
	  eta_old[i+j+modes] = q_dot[j][l];
  }
	  l += 1;
  } 
//
  vector<vector<su2double> > ASys(NT*dofs,vector<su2double>(NT*dofs+1,0.0));
  su2double dtau = config->GetPseudoTimeStep(); 

  cout << "Solving System..." << endl; 
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {

         ASys[i][j] = dtau*(AHB[i][j] + KHB[i][j]);
         if (i==j) ASys[i][i] +=  1.0;

      }      
  }

  for(int j=0;j<NT*dofs;j++) {

        ASys[j][NT*dofs] = eta_old[j] +  dtau*Force[j];

  }

  Gauss_Elimination(ASys, eta);

////cout << "eta_old" << endl;
////for(int j=0;j<NT*dofs;j++) { 
////         cout << eta_old[j] << "\n";
////}
////cout << endl;

////cout << "eta" << endl;
////for(int j=0;j<NT*dofs;j++) { 
////         cout << eta[j] << "\n";
////}
////cout << endl;

  l = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {
  for(int j=0;j<modes;j++) {

	  q[j][l]     = eta[i+j];
	  q_dot[j][l] = eta[i+j+modes];
  }
	  l += 1;
  } 
 

  //

////cout << "q" << endl;
////  for(int j=0;j<NT;j++) { 
////         cout << q[0][j]  << "\n";
////}
////  cout << endl;

////  cout << "q DOT" << endl;
////  for(int j=0;j<NT;j++) { 
////         cout << q_dot[0][j]  << "\n";
////}
////  cout << endl;
  
  
//
// PHASE FIX
// 
  vector<vector<su2double>> qh(modes,vector<su2double>(NT,0.0));
  vector<vector<su2double>> qhd(modes,vector<su2double>(NT,0.0));

  for(int k=0;k<modes;k++) { 
  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       qh[k][i] += EE[i][j]*q[k][j];
  
       qhd[k][i] += EE[i][j]*q_dot[k][j];
       
  }
  }
  }

////////////

            int fixmode = 0;
            su2double U0  = qh[fixmode][0], Cc1 = qh[fixmode][1], Cs1 = qh[fixmode][2]; ///// Cc2 = ah[3], Cs2 = ah[4];

            su2double magn  = sqrt(Cc1*Cc1 + Cs1*Cs1);
            su2double ratio = config->GetMode_Ampl(fixmode)/magn;

	    cout << "Ratio= " << ratio << endl;
            /////if (ratio < 0.9) { ratio = 0.9;}
 
///            U0  = qh[fixmode][0], Cc1 = qh[fixmode][1], Cs1 = qh[fixmode][2];

            su2double phase = atan2(Cs1,Cc1) - PI_NUMBER/2;
    
            su2double dt_lag = phase/OmegaHB;
            cout << "phase= " << phase << endl;

            vector<su2double> time_hb(NT,0.0);
            for(int i=0;i<NT;i++) {

                    time_hb[i] = i*Period/NT + dt_lag;
            }

   	    if (config->HB_Flutter()) { 

	    	    //if (ratio < 1.0) { 
              
	  		    //qh[fixmode][0]  = qh[fixmode][0]*ratio;      
		    	    //qhd[fixmode][0] = qhd[fixmode][0]*ratio;
			    //qh[fixmode][0]  = 0.0;
			    //qhd[fixmode][0] = 0.0;
		    //}
		    for (int i=1;i<NT;i++) {
          
			    qh[fixmode][i]  = qh[fixmode][i]*ratio;
			    qhd[fixmode][i] = qhd[fixmode][i]*ratio;
		   }

		  //for (int j=0;j<modes;j++) {
		  //for (int i=1;i<NT;i++) {
          
		  //        qh[j][i]  = qh[j][i]*ratio;
		  //        qhd[j][i] = qhd[j][i]*ratio;
		  //}
		  //}
	    }


  cout << "a_hat  |  h_hat" << endl;
  for(int i=0;i<NT;i++) { 

      cout << qh[0][i] << " | " << "\n";

  }
  cout << endl;
 
            for(int k=0;k<modes;k++) {
            for(int i=0;i<NT;i++) {

                    q[k][i]     = qh[k][0];
                    q_dot[k][i] = qhd[k][0];

                    for(int j=0;j<NH;j++) {

                    q[k][i]     = q[k][i] + qh[k][2*j+1]*cos((j+1)*OmegaHB*time_hb[i]) 
                	        + qh[k][2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);
                   
                    q_dot[k][i] = q_dot[k][i] + qhd[k][2*j+1]*cos((j+1)*OmegaHB*time_hb[i]) 
                	        + qhd[k][2*j+2]*sin((j+1)*OmegaHB*time_hb[i]);
          
                    }

            }
            }

 /// }
//cout << "a_hat  |  h_hat" << endl;
//for(int i=0;i<NT;i++) { 

//    cout << qh[0][i] << " | " << "\n";

//}
//cout << endl;

//cout << "a  |  h" << endl;
//for(int i=0;i<NT;i++) { 

//    cout << q[0][i] << " | "  << "\n";

//}
//cout << endl;


   /*--- Set the solution of the structural equations ---*/

  for (int i=0;i<NT;i++) {
  for (int j=0;j<modes;j++) {

  gen_displacements[i][j] = q[j][i];

  }
  }

  for (int i=0;i<NT;i++){  
  for (int j=0;j<modes;j++) {
     config->SetHB_Modal_Displacement(q[j][i], i, j);
     config->SetHB_Modal_Velocities(q_dot[j][i], i, j);
  }
  }

  for (int i=0;i<NT;i++){  
    config->SetHB_pitch(q[0][i], i);
    config->SetHB_plunge(q[1][i], i);
////config->SetHB_pitch_rate(q[2][i], i);
////config->SetHB_plunge_rate(q[3][i], i);
  }


}

void CSolver::Frequency_Update_Phy(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 4;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency" << endl;
  //
  vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  Kdofs[0][2] = -1.0;
  Kdofs[1][3] = -1.0;

  Kdofs[2][0] = AK[0][0];
  Kdofs[2][1] = AK[0][1];
  Kdofs[3][0] = AK[1][0];
  Kdofs[3][1] = AK[1][1];

  Kdofs[2][2] = AT[0][0];
  Kdofs[2][3] = AT[0][1];
  Kdofs[3][2] = AT[1][0];
  Kdofs[3][3] = AT[1][1];

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = Kdofs[j][k];
	  }
	  }
	
  }  
  
  vector<su2double> Force(NT*dofs,0);
 
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta_dot(2,vector<su2double>(NT,0.0)); 

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
    q_dot[1][i] = config->GetHB_pitch_rate(i)/w_alpha;
    q_dot[0][i] = config->GetHB_plunge_rate(i)/b/w_alpha;
  }

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta[i+0] = q[0][k];
          eta[i+1] = q[1][k];

          eta[i+2] = q_dot[0][k];
          eta[i+3] = q_dot[1][k];

          k += 1;
  }
  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;


//for(int i=0;i<NT*dofs;i++) {

//      eta[i] = config->GetHB_displacements(i);

//}
//
//  if (TimeIter % 5 == 0){
//for(int i=0;i<NT*dofs;i++) {

//    config->SetHB_displacements(eta[i], i);

//}
//  }
  //
  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

               cout << "Lift= " << cl[i] << " Moment= " << cm[i] << "\n" << endl; 
  }

//for(int i=0;i<NT*dofs;i++) {

//    Force[i] = config->GetHB_forces(i);

//}

  for(int count=0;count<NT;count++){

  f[0] = cons*(-cl[count]);
  f[1] = cons*(2*-cm[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += M_inv[i][k]*f[k]; //PHI transpose
    }
  }

  Force[count*dofs+2] = f_tilde[0];
  Force[count*dofs+3] = f_tilde[1];

  }
  //
//for (int i=0;i<NT*dofs;i++) {
//for (int j=0;j<NT*dofs;j++) {

//        DM[i][j] = AHB[i][j]/OmegaHB;

//}
//}
  su2double L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
	  
	 num1[i] = -Force[i];
         Rs[i]   = -Force[i];
         num2[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

	 Rs[i]   += (OmegaHB / w_a * DM[i][j] + KHB[i][j])*eta[j];
	 num1[i] += KHB[i][j] * eta[j]; 
	 num2[i] += DM[i][j] * eta[j];

  }

  num -= (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += 0.5*(Rs[i]*Rs[i]); 

  } 

  //su2double OmegaHBnew = num/den;
  su2double OmegaHBcheck = num/den;
  su2double OmegaHBnew = w_a*OmegaHBcheck;
  su2double PeriodHBnew = 2*PI_NUMBER/OmegaHBnew;
  su2double error_value = (OmegaHBnew - OmegaHB)/OmegaHB;


  cout << "New HB Omega = " << (OmegaHBcheck) << " / L2NORM = " << L2norm << endl;
  cout << "Error in omega calc = " << error_value << endl;

  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);
  config->SetStr_L2_norm(L2norm);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::Frequency_Update_2D(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = AK[j][k];
	  }
	  }
	
  }  
  
  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta_dot(2,vector<su2double>(NT,0.0)); 

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
 }

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta[i+0] = q[0][k];
          eta[i+1] = q[1][k];

          k += 1;
  }

//for(int i=0;i<NT*dofs;i++) {

//    eta[i] = config->GetHB_displacements(i);

//}

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

               cout << "Lift= " << cl[i] << " Moment= " << cm[i] << "\n" << endl; 
  }

//for(int i=0;i<NT*dofs;i++) {

//    Force[i] = config->GetHB_forces(i);

//}

  for(int count=0;count<NT;count++){

  f[0] = (-cl[count]);
  f[1] = (2*-cm[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += M_inv[i][k]*f[k]; //PHI transpose
    }
  }

  Force[count*dofs+0] = f_tilde[0];
  Force[count*dofs+1] = f_tilde[1];

  }
//
//////////////////////////////////////////////////////////////////////////
//
  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double vf_old = vf;
  su2double OmegaHBcheck = OmegaHB*OmegaHB/w_a /w_a;
  su2double OmegaHBnew;
  su2double PeriodHBnew;
  su2double error_value;


  cons = vf*vf/PI_NUMBER;
  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = -cons*Force[i];
         Rs[i]   = -cons*Force[i];
         num2[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ((OmegaHBcheck * DM2[i][j] + KHB[i][j])*eta[j]);
         num1[i] += (KHB[i][j] * eta[j]); 
         num2[i] += (DM2[i][j] * eta[j]);

  }

  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  OmegaHBcheck = -num/den;
  OmegaHBnew   = w_a*sqrt(OmegaHBcheck);
  error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;

  if (OmegaHBcheck <= 0.0) { cout << "Neg. Frequency" << endl; OmegaHBnew = OmegaHB; };

  OmegaHBnew = OmegaHB + 0.3*(OmegaHBnew - OmegaHB);
  PeriodHBnew  = 2*PI_NUMBER/OmegaHBnew;
  
//  OmegaHBnew = OmegaHB + (OmegaHBnew - OmegaHB);
  //OmegaHBnew = 65; 
////////////////////////////////////////////////////////////////////
//
  cout << "New HB Omega = " << (OmegaHBnew) << " / L2NORM = " << L2norm  << endl;
  cout << "Error in omega calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);
  config->SetStr_L2_norm(L2norm);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::Frequency_Update_3D(CConfig *config, su2double**& gen_forces, int harmonics, bool &error_flag) {

 
  /*--- Retrieve values from the config file ---*/
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> w_modes(modes,0.02); //contains solution(displacements and rates) of typical section wing model.

  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetRefWing_Length()/2.0; // root airfoil semichord
  su2double Vo      = config->GetConicalRefVol();
  su2double scale_param = 1/config->Get_Scaling_Parameter();

  su2double OmegaHB_new, PeriodHB_new;
  su2double L2norm;
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
 
  su2double DEG2RAD = PI_NUMBER/180.0;
  su2double w_alpha = w_a;

  int dofs = modes, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(modes,0.02);
  for (unsigned short i=0;i<modes; i++) {

          w_modes[i] = config->GetAero_Omega(i)/w_a;

  }

  su2double num = 0.0;
  su2double den = 0.0;

  if (rank == MASTER_NODE)
  { 
  cout << "Updating Frequency (3D)" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
  vector<vector<su2double> > K(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          K[i][i] = w_modes[i]*w_modes[i];

  }

   // Stiffness Matrix
  vector<vector<su2double> > T(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          T[i][i] = 2.0*xi[i]*w_modes[i];

  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DT(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
  }

  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> THB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  THB[i*dofs+j][i*dofs+k] = T[j][k];
	  }
	  }
	
  }  
    for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DT[i][j] += THB[i][k] * DM[k][j];
	  }
      }
  }


  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf*b*b/2.0/Vo;
  vector<su2double> eta(NT*dofs,0.0);
  vector<vector<su2double>> q(modes,vector<su2double>(NT,0.0));
// 
  cout << "Computing Forces..." << endl;
  for(int count=0;count<NT;count++){
  for (int i=0; i<modes; i++) {
   
    Force[count*dofs+i] = (cons * scale_param *gen_forces[count][i]);
    
  }
  }
	
  for (int i=0;i<NT;i++){  
  for (int j=0;j<modes;j++) {
    q[j][i]     = config->GetHB_Modal_Displacement(i, j);
  }
  }  

  int l = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {
  for(int j=0;j<modes;j++) {

	  eta[i+j] = q[j][l];
  }
	  l += 1;
  }
//

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;
 
//////////////////////////////////////////////////////////////////////////

  su2double Omega  = OmegaHB/w_a;
  su2double Omega2 = OmegaHB*OmegaHB/w_a/w_a;
  su2double Omega3 = OmegaHB*OmegaHB*OmegaHB/w_a/w_a/w_a;
  su2double Omega_new = Omega, Omega_old = Omega;

  su2double error_value, NR_error;
  su2double zeta, dzeta, relax = 0.3;
  su2double k0, k1, k2, k3;

  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> num3(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = -Force[i];
         Rs[i]   = -Force[i];
         num2[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ((Omega2 * DM2[i][j] + KHB[i][j])*eta[j]);
         num1[i] += (KHB[i][j] * eta[j]); 
         num2[i] += (DM2[i][j] * eta[j]);

  }

  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  su2double OmegaHBcheck = -num/den;
  OmegaHB_new   = w_a*sqrt(OmegaHBcheck);
  error_value  = (OmegaHB_new - OmegaHB)/OmegaHB;

  ///OmegaHB_new  = OmegaHB + 0.7*(OmegaHB_new - OmegaHB);

  if (OmegaHBcheck <= 0.0) { cout << "Neg. Frequency" << endl; OmegaHB_new = OmegaHB; };

//
////////////////////////////////////////////////////////////////////
//
//cout << "Omega old=" << Omega_old << endl; 
//cout << "Starting NR iterations..." <<  endl;
//for (int IJK=0;IJK<1;IJK++) {
////
////
//L2norm = 0.0;
//k0 = 0.0;
//k1 = 0.0;
//k2 = 0.0;
//k3 = 0.0;
//for (int i=0;i<NT*dofs;i++) {
//        
//       Rs[i]   = -cons*Force[i];
//       num1[i] = 0.0;
//       num2[i] = 0.0;
//       num3[i] = 0.0;

//for (int j=0;j<NT*dofs;j++) {

//       Rs[i]   += ( Omega2 * DM2[i][j] * eta[j] 
//                  + Omega * DT[i][j] * eta[j] 
//                  + KHB[i][j] * eta[j] );

//       num1[i] += (DM2[i][j] * eta[j]);
//       num2[i] += (DT[i][j]  * eta[j]);
//       num3[i] += (KHB[i][j] * eta[j]); 
//}

//k0 += (num3[i] * num2[i] - Force[i] * num2[i]);
//k1 += (2.0 * num1[i] * num1[i]);
//k2 += (3.0 * num2[i] * num1[i]); 
//k3 += (num2[i] * num2[i] + 2.0 * num3[i] * num1[i] - Force[i] * num1[i]);

//L2norm += (0.5*(Rs[i]*Rs[i])); 

//} 

//zeta  = k3 * Omega3 + k2 * Omega2 + k1 * Omega + k0;
//dzeta = 3.0 * k3 * Omega2 + 2.0 * k2 * Omega + k1;

//Omega_new = Omega_old - relax * zeta / dzeta;

//NR_error = (Omega_new - Omega_old)/Omega_old;

//Omega_old = Omega_new;
//Omega     = Omega_new;
//Omega2    = Omega * Omega;
//Omega3    = Omega * Omega * Omega;

//cout << "Omega_new= " << Omega_new << endl;
//}
//cout << "Freq. Prediction: NR error= " << NR_error <<  endl;

//OmegaHB_new   = w_a*sqrt(Omega2);
//error_value  = (OmegaHB_new - OmegaHB)/OmegaHB;

//if (OmegaHB_new < 0.0) OmegaHB_new = OmegaHB;

//PeriodHB_new  = 2*PI_NUMBER/OmegaHB_new;
//
//////////////////////////////////////////////////////////////////////////////////////

  cout << "New HB Omega = " << (OmegaHB_new) << " / L2NORM = " << L2norm  << endl;
  cout << "Error in omega calc = " << error_value << endl;
 
  if ((abs(error_value) < 0.0000001 || OmegaHB_new < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 
       	
  for (int destination=1;destination<SU2_MPI::GetSize();destination++){
	      SU2_MPI::Send(&OmegaHB_new, 1, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
	      SU2_MPI::Send(&L2norm, 1, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
  }
         
  }  
  else {
	  SU2_MPI::Recv(&OmegaHB_new, 1, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
	  SU2_MPI::Recv(&L2norm, 1, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
  }
          
  SU2_MPI::Barrier(SU2_MPI::GetComm());

  PeriodHB_new  = 2*PI_NUMBER/OmegaHB_new;
  config->SetHarmonicBalance_Period(PeriodHB_new);
  config->SetStr_L2_norm(L2norm);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHB_new, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHB_new, NH + i);

}

void CSolver::Velocity_Update_2D(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = AK[j][k];
	  }
	  }
	
  }  
  
  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta_dot(2,vector<su2double>(NT,0.0)); 

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
 }

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta[i+0] = q[0][k];
          eta[i+1] = q[1][k];

          k += 1;
  }

//for(int i=0;i<NT*dofs;i++) {

//    eta[i] = config->GetHB_displacements(i);

//}

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

               cout << "Lift= " << cl[i] << " Moment= " << cm[i] << "\n" << endl; 
  }

//for(int i=0;i<NT*dofs;i++) {

//    Force[i] = config->GetHB_forces(i);

//}

  for(int count=0;count<NT;count++){

  f[0] = (-cl[count]);
  f[1] = (2*-cm[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += M_inv[i][k]*f[k]; //PHI transpose
    }
  }

  Force[count*dofs+0] = f_tilde[0];
  Force[count*dofs+1] = f_tilde[1];

  }

//////////////////////////////////////////////////////////////////////////

  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double vf_old = vf, vf2 = vf*vf, vf_new;
  su2double OmegaHBcheck = OmegaHB*OmegaHB/w_a /w_a;
  su2double OmegaHBnew;
  su2double PeriodHBnew;
  su2double error_value, NR_error, zeta, dzeta;
  su2double gamma = 0.00000001;

  cons = 1.0/PI_NUMBER;

  for (int IJK=0;IJK<1;IJK++) {
  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num2[i] = -cons*Force[i];
         Rs[i]   = -cons*vf2*Force[i];
         num1[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ((OmegaHBcheck * DM2[i][j] + KHB[i][j])*eta[j]);
         num1[i] += (OmegaHBcheck * DM2[i][j] * eta[j] + KHB[i][j] * eta[j]); 
         //num2[i] += (DM2[i][j] * eta[j]);

  }

  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  gamma = L2norm;

  zeta  = den*vf2*vf2*vf + num*vf2*vf - gamma;
  dzeta = 5*den*vf2*vf + 3*num*vf2;

  vf_new = vf_old - 0.7*zeta/dzeta;

//vf2 = -num/den;
//vf  = sqrt(vf2);
  NR_error = (vf_new - vf_old)/vf_old;

  vf = vf_new;
  vf_old = vf_new;
  vf2 = vf*vf;
  }
  cout << "N-R error = " << NR_error << endl;
 
  error_value = (vf_new-vf)/vf;
  //vf = vf_new - 0.7*(vf_new-vf);
 

//if (TimeIter == 1) {
//  vf = 0.69;
//}
//else if (TimeIter == 2) {
//  vf = 0.68;
//}
//else if (TimeIter == 3) {
//  vf = 0.689;
//}
//else if (TimeIter == 15) {
//  vf = 0.688;
//}
//else if (TimeIter == 35) {
//  vf = 0.687;
//}
//else if (TimeIter > 3 && TimeIter < 15) {
//  vf = 0.689;
//}
//else if (TimeIter > 15 && TimeIter < 35) {
//  vf = 0.688;
//}
//else vf = 0.687;

////////////////////////////////////////////////////////////////////
//
  cout << "New  VF = " << vf << " / L2NORM = " << L2norm  << endl;
  cout << "Error in velocity calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || vf < 0.01)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(vf);
  
}

void CSolver::Velocity_Update_3D(CConfig *config, su2double**& gen_forces, int harmonics, bool &error_flag) {
 
  /*--- Retrieve values from the config file ---*/
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> w_modes(modes,0.0); //contains solution(displacements and rates) of typical section wing model.

  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetRefWing_Length()/2.0; // root airfoil semichord
  su2double Vo      = config->GetConicalRefVol();
  su2double scale_param = 1/config->Get_Scaling_Parameter();

  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
 
  su2double L2norm, FlutterSpeed;
  su2double DEG2RAD = PI_NUMBER/180.0;
  su2double w_alpha = w_a;

  int dofs = modes, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(modes,0.02);
  for (unsigned short i=0;i<modes; i++) {

          w_modes[i] = config->GetAero_Omega(i)/w_a;

  }

  su2double num = 0.0;
  su2double den = 0.0;

  if (rank == MASTER_NODE)
  { 
//
  cout << "Updating Velocity (3D)" << endl;
//
  vector<vector<su2double> > K(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          K[i][i] = w_modes[i]*w_modes[i];

  }

   // Stiffness Matrix
  vector<vector<su2double> > T(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          T[i][i] = 2.0*xi[i]*w_modes[i];

  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> DT(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> THB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  THB[i*dofs+j][i*dofs+k] = T[j][k];
	  }
	  }
	
  }  
    for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DT[i][j] += THB[i][k] * DM[k][j];
	  }
      }
  }

 
  vector<su2double> Force(NT*dofs,0);
  su2double cons = b*b/Vo/2.0;
  vector<su2double> eta(NT*dofs,0.0);
  vector<vector<su2double>> q(modes,vector<su2double>(NT,0.0));
// 
  cout << "Computing Forces..." << endl;
  for(int count=0;count<NT;count++){
  for (int i=0; i<modes; i++) {
   
    Force[count*dofs+i] = (cons * scale_param * gen_forces[count][i]);
    
  }
  }
	
  for (int i=0;i<NT;i++){  
  for (int j=0;j<modes;j++) {
    q[j][i]     = config->GetHB_Modal_Displacement(i, j);
  }
  }  

  int l = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {
  for(int j=0;j<modes;j++) {

	  eta[i+j] = q[j][l];
  }
	  l += 1;
  }
//

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;
 
//////////////////////////////////////////////////////////////////////////

  su2double FS2;

  su2double vf_old = vf, vf2 = vf*vf, vf_new;
  su2double Omega2 = OmegaHB*OmegaHB/w_a /w_a;
  su2double OmegaHBnew;
  su2double PeriodHBnew;
  su2double error_value, NR_error, zeta, dzeta;
  su2double gamma = 0.00000001;

  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> num3(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         Rs[i]   = -vf*vf*Force[i];
         num1[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ((Omega2 * DM2[i][j] + KHB[i][j])*eta[j]);
         num1[i] += ((Omega2 * DM2[i][j] + KHB[i][j])*eta[j]); 

  }

  num += (Force[i] * num1[i]);
  den += (Force[i] * Force[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  FS2     = num/den;
  vf_new  = sqrt(FS2);

  error_value  = (vf_new - vf)/vf;

  su2double eta_relax = 1.0;

  FlutterSpeed = vf + eta_relax*(vf_new - vf);

  ///vf_new = vf + 0.7*(vf_new-vf);

//for (int IJK=0;IJK<1;IJK++) {
//L2norm = 0.0;
//num = 0.0;
//den = 0.0;
//for (int i=0;i<NT*dofs;i++) {
//        
//       num2[i] = -Force[i];
//       Rs[i]   = -vf2*Force[i];
//       num1[i] = 0.0;

//for (int j=0;j<NT*dofs;j++) {

//       Rs[i]   += (Omega2 * DM2[i][j] * eta[j] + KHB[i][j] * eta[j]);
//       num1[i] += (Omega2 * DM2[i][j] * eta[j] + KHB[i][j] * eta[j]); 
//       //num2[i] += (DM2[i][j] * eta[j]);
//}

//num += (num1[i] * num2[i]);
//den += (num2[i] * num2[i]); 

//L2norm += (0.5*(Rs[i]*Rs[i])); 

//} 

//gamma = L2norm;

//zeta  = den*vf2*vf2*vf + num*vf2*vf - gamma;
//dzeta = 5*den*vf2*vf + 3*num*vf2;

//vf_new = vf_old - 0.7*zeta/dzeta;

//NR_error = (vf_new - vf_old)/vf_old;

//vf = vf_new;
//vf_old = vf_new;
//vf2 = vf*vf;
//}
//cout << "N-R error = " << NR_error << endl;

//error_value = (vf_new-vf)/vf;
////vf = vf_new - 0.7*(vf_new-vf);

////////////////////////////////////////////////////////////////////

  cout << "New  VF = " << vf_new << " / L2NORM = " << L2norm  << endl;
  cout << "Error in velocity calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || vf < 0.01)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  for (int destination=1;destination<SU2_MPI::GetSize();destination++) {
	      SU2_MPI::Send(&FlutterSpeed, 1, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
	      SU2_MPI::Send(&L2norm, 1, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
  }
         
  }  
  else {
	  SU2_MPI::Recv(&FlutterSpeed, 1, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
	  SU2_MPI::Recv(&L2norm, 1, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
  }

  SU2_MPI::Barrier(SU2_MPI::GetComm());
  config->SetStr_L2_norm(L2norm);
  config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
  
}

void CSolver::Frequency_Velocity_Update_3D(CConfig *config, su2double**& gen_forces, int harmonics, bool &error_flag) {
 
  /*--- Retrieve values from the config file ---*/
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> w_modes(modes,0.0); //contains solution(displacements and rates) of typical section wing model.

  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetRefWing_Length()/2.0; // root airfoil semichord
  su2double Vo      = config->GetConicalRefVol();
  su2double scale_param = 1/config->Get_Scaling_Parameter();

  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
 
  su2double DEG2RAD = PI_NUMBER/180.0;
  su2double w_alpha = w_a;

  int dofs = modes, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(modes,0.02);
  for (unsigned short i=0;i<modes; i++) {

          w_modes[i] = config->GetAero_Omega(i)/w_a;

  }

  su2double num = 0.0;
  su2double den = 0.0;

//
  cout << "Updating Frequency & Velocity (3D)" << endl;
//
  vector<vector<su2double> > K(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          K[i][i] = w_modes[i]*w_modes[i];

  }

   // Stiffness Matrix
  vector<vector<su2double> > T(modes,vector<su2double>(modes,0.0));
  for (unsigned short i=0;i<modes; i++) {

          T[i][i] = 2.0*xi[i]*w_modes[i];

  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> DT(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> THB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  THB[i*dofs+j][i*dofs+k] = T[j][k];
	  }
	  }
	
  }  
    for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DT[i][j] += THB[i][k] * DM[k][j];
	  }
      }
  }

 
  vector<su2double> Force(NT*dofs,0);
  su2double cons = b*b/Vo/2.0;
  vector<su2double> eta(NT*dofs,0.0);
  vector<vector<su2double>> q(modes,vector<su2double>(NT,0.0));
// 
  cout << "Computing Forces..." << endl;
  for(int count=0;count<NT;count++){
  for (int i=0; i<modes; i++) {
   
    Force[count*dofs+i] = (cons * scale_param * gen_forces[count][i]);
    
  }
  }
	
  for (int i=0;i<NT;i++){  
  for (int j=0;j<modes;j++) {
    q[j][i]     = config->GetHB_Modal_Displacement(i, j);
  }
  }  

  int l = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {
  for(int j=0;j<modes;j++) {

	  eta[i+j] = q[j][l];
  }
	  l += 1;
  }
//

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;
 
//////////////////////////////////////////////////////////////////////////

  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double Omega  = OmegaHB/w_a;
  su2double Omega2 = OmegaHB*OmegaHB/w_a/w_a;
  su2double Omega3 = OmegaHB*OmegaHB*OmegaHB/w_a/w_a/w_a;
  su2double Omega_new = Omega, Omega_old = Omega;
  su2double OmegaHB_new, PeriodHB_new;

  su2double vf_old = vf, vf2 = vf*vf, vf_new;
  su2double error_vf_value, error_w_value, error_value;

  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> num3(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);

  for (int IKL=0;IKL<1;IKL++) {

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         num1[i] += ((Omega2 * DM2[i][j] + KHB[i][j])*eta[j]); 

  }

  num += (Force[i] * num1[i]);
  den += (Force[i] * Force[i]); 

  } 

  vf2     = num/den;
  vf_new  = sqrt(vf2);

  error_vf_value  = (vf_new - vf)/vf;

  FlutterSpeed = vf_new;

  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = -vf2*Force[i];
         Rs[i]   = -vf2*Force[i];
         num2[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ((Omega2 * DM2[i][j] + KHB[i][j])*eta[j]);
         num1[i] += (KHB[i][j] * eta[j]); 
         num2[i] += (DM2[i][j] * eta[j]);

  }

  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  Omega2       = -num/den;
  Omega        = sqrt(Omega2);
  OmegaHB_new  = w_a * Omega;
  PeriodHB_new = 2*PI_NUMBER/OmegaHB_new;

  error_w_value  = (OmegaHB_new - OmegaHB)/OmegaHB;

  }

  error_value = error_vf_value + error_w_value;
///////////////////////////////////////////////////////////////////////////////////

  cout << "New  VF = " << vf_new << " / L2NORM = " << L2norm  << endl;
  cout << "Error in velocity calc = " << error_vf_value << endl;
  cout << "New HB Omega = " << (OmegaHB_new) << " / L2NORM = " << L2norm  << endl;
  cout << "Error in omega calc = " << error_w_value << endl;
  
///////////////////////////////////////////////////////////////////////////////////

  if ((abs(error_value) < 0.000001 || vf < 0.01)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(vf_new);

  config->SetHarmonicBalance_Period(PeriodHB_new);
  config->SetStr_L2_norm(L2norm);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHB_new, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHB_new, NH + i);
 
}

void CSolver::Velocity_Update_2D2(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 4;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  su2double num = 0.0;
  su2double den = 0.0;
  su2double lum = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency" << endl;
  //
  vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  Kdofs[0][2] = -1.0;
  Kdofs[1][3] = -1.0;

  Kdofs[2][0] = AK[0][0];
  Kdofs[2][1] = AK[0][1];
  Kdofs[3][0] = AK[1][0];
  Kdofs[3][1] = AK[1][1];

  Kdofs[2][2] = AT[0][0];
  Kdofs[2][3] = AT[0][1];
  Kdofs[3][2] = AT[1][0];
  Kdofs[3][3] = AT[1][1];


  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator_Complex(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = Kdofs[j][k];
	  }
	  }
	
  }  
  
  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta_dot(2,vector<su2double>(NT,0.0)); 

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
    q_dot[1][i] = config->GetHB_pitch_rate(i)/w_alpha;
    q_dot[0][i] = config->GetHB_plunge_rate(i)/b/w_alpha;
 }

  vector<su2double> pl_h(NT,0.0);
  vector<su2double> pi_h(NT,0.0);
  vector<su2double> plr_h(NT,0.0);
  vector<su2double> pir_h(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       pl_h[i] += EE[i][j]*q[0][j];
       pi_h[i] += EE[i][j]*q[1][j];

       plr_h[i] += EE[i][j]*q_dot[0][j];
       pir_h[i] += EE[i][j]*q_dot[1][j];  
  }
  }

//int k = 0;
//for(int i=0;i<NT*dofs;i+=dofs) {

//        eta[i+0] = q[0][k];
//        eta[i+1] = q[1][k];

//        eta[i+2] = q_dot[0][k];
//        eta[i+3] = q_dot[1][k];

//        k += 1;
//}

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta[i+0] = pl_h[k];
          eta[i+1] = pi_h[k];

          eta[i+2] = plr_h[k];
          eta[i+3] = pir_h[k];

          k += 1;
  }

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

               cout << "Lift= " << cl[i] << " Moment= " << cm[i] << "\n" << endl; 
  }

  vector<su2double> clh(NT,0.0);
  vector<su2double> cmh(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       clh[i] += EE[i][j]*cl[j];
       cmh[i] += EE[i][j]*cm[j];
        
  }
  }

  for(int count=0;count<NT;count++){

  f[0] = (-clh[count]);
  f[1] = (2*-cmh[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += M_inv[i][k]*f[k]; //PHI transpose
    }
  }

  Force[count*dofs+2] = f_tilde[0];
  Force[count*dofs+3] = f_tilde[1];

  }

//////////////////////////////////////////////////////////////////////////

  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double vf_old = vf, vf2 = vf*vf, vf_new, ovf2 = 1.0/vf/vf, ovf3=ovf2/vf;
  su2double OmegaHBcheck = OmegaHB/w_a;
  su2double OmegaHBnew;
  su2double PeriodHBnew;
  su2double error_value, NR_error, zeta, dzeta;
  su2double gamma = 0.00000001;

  cons = 1.0/PI_NUMBER;

  for (int IJK=0;IJK<1;IJK++) {

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  lum = 0.0;
  for (int i=4;i<NT*dofs;i++) {
          
         num2[i] = -cons*Force[i];
         Rs[i]   = -cons*vf2*Force[i];
         num1[i] = 0.0;

  for (int j=4;j<NT*dofs;j++) {

         Rs[i]   += ((OmegaHBcheck * DM[i][j] + KHB[i][j])*eta[j]);
         num1[i] += (OmegaHBcheck * DM[i][j] * eta[j] + KHB[i][j] * eta[j]); 

  }

  num += (num1[i] * num1[i]);
  lum += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  cout << "num= " << num << " | den= " << den << " | lum= " << lum << endl;

  zeta  =  -num*ovf2 + 3.0*den*vf2 + 2.0*lum;
  dzeta = 2*num*ovf3 + 6*den*vf;
//zeta  =  -num + 3.0*den*vf2*vf2 + 2.0*lum*vf2;
//dzeta =  12.0*den*vf2*vf + 4.0*lum*vf;

  cout << "zeta= " << zeta << " | dzeta= " << dzeta << endl;

  vf_new = vf_old - 0.3*zeta/dzeta;

  NR_error = (vf_new - vf_old)/vf_old;

  vf_old = vf_new;

  vf   = vf_new;
  vf2  = vf*vf; 
  ovf2 = 1.0/vf/vf;
  ovf3 = ovf2/vf;

  }
  cout << "N-R error = " << NR_error << endl;
 
  error_value = (vf_new-vf)/vf;
////////////////////////////////////////////////////////////////////

  cout << "New  VF = " << vf << " / L2NORM = " << L2norm  << endl;
  cout << "Error in velocity calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || vf < 0.01)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(vf);
  
}

void CSolver::Frequency_Velocity_Update_2D(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double mu  = config->GetAeroelastic_Airfoil_Mass_Ratio();

  su2double Uinf = config->GetVelocity_FreeStream()[0];
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);
//x[0] = 0.00038;
//x[1] = 0.00013;
//
  cout << "Updating Frequency & Velocity" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  vector<vector<su2double> > AK(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > AT(2,vector<su2double>(2,0.0));
 
  for (int i=0; i<2; i++) {
  for (int j=0; j<2; j++) {
    for (int k=0; k<2; k++) {
      AK[i][j] += M_inv[i][k]* K[k][j];
      AT[i][j] += M_inv[i][k]* T[k][j];
    }  
  }
  }

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  
  HB_Operator(config, DD, EE, NT, OmegaHB);

   for (int i=0;i<dofs;i++) {
   for (int j=0;j<NT;j++)   {
   for (int k=0;k<NT;k++)   {

        DHB[i*NT+j][i*NT+k] = DD[j][k];
   
   }
   }
   }

//
// transform for dofs
//
  int position = 0;
  vector<vector<su2double>> Q(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> Qt(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for (int i=0; i<dofs; i++) {
	for (int j=0; j<NT; j++) {
		Q[NT*i+j][position+j*dofs] = 1;
	}
	position = position + 1;
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      Qt[j][i] = Q[i][j];
    }
  }
//  
  vector<vector<su2double>> DM(NT*dofs,vector<su2double>(NT*dofs,0.0));
  vector<vector<su2double>> DM2(NT*dofs,vector<su2double>(NT*dofs,0.0)); 
  vector<vector<su2double>> tmp(NT*dofs,vector<su2double>(NT*dofs,0.0));  
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        tmp[i][j] += DHB[i][k] * Q[k][j];
      }
    }
  }
  for(int i=0;i<NT*dofs;i++) {
    for(int j=0;j<NT*dofs;j++) {
      for(int k=0;k<NT*dofs;k++) {
        DM[i][j] += Qt[i][k] * tmp[k][j];
      }
      //DM[i][j] = DM[i][j] ;
    }
  } 

  for(int i=0;i<NT*dofs;i++) {
      for(int j=0;j<NT*dofs;j++) {
	  for(int k=0;k<NT*dofs;k++) {
             DM2[i][j] += DM[i][k] * DM[k][j];
	  }
      }
     // AHB[i][j] = ( AHB[i][j] * OmegaHB ) / w_alpha;
    }


  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = AK[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> THB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  THB[i*dofs+j][i*dofs+k] = AT[j][k];
	  }
	  }
	
  }  

  vector<su2double> Force(NT*dofs,0);
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  vector<su2double> f_tilde(2,0.0);
  vector<su2double> eta(NT*dofs,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> q_dot(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta(2,vector<su2double>(NT,0.0));
//  vector<vector<su2double>> theta_dot(2,vector<su2double>(NT,0.0)); 

  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
 }

  int k = 0;
  for(int i=0;i<NT*dofs;i+=dofs) {

          eta[i+0] = q[0][k];
          eta[i+1] = q[1][k];

          k += 1;
  }

//for(int i=0;i<NT*dofs;i++) {

//    eta[i] = config->GetHB_displacements(i);

//}

  cout << "ETA" << endl;
  for (int i=0;i<NT*dofs;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

  }

//for(int i=0;i<NT*dofs;i++) {

//    Force[i] = config->GetHB_forces(i);

//}

  cout << "LIFT    /    MOMENT" << endl;
  for(int count=0;count<NT;count++){

               cout  << cl[count] << " " << cm[count] << "\n" << endl; 

  f[0] = (-cl[count]);
  f[1] = (2*-cm[count]);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += (M_inv[i][k]*f[k]); //PHI transpose
    }
  }

  Force[count*dofs+0] = f_tilde[0];
  Force[count*dofs+1] = f_tilde[1];

  }
//
//////////////////////////////////////////////////////////////////////////
//
  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double vf_old = vf, vf_new;

  su2double w_t  = OmegaHB/w_a;
  su2double w2_t = w_t*w_t, vf2 = vf*vf;

  su2double OmegaHBcheck;
  su2double OmegaHBnew;
  su2double PeriodHBnew;
  su2double error_value, error_value_vf;

  for (int IKL=0;IKL<1;IKL++) {

  cons = 1.0/PI_NUMBER;
  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = -cons*vf2*Force[i];
         Rs[i]   = -cons*vf2*Force[i];
         num2[i] = 0.0;

  for (int j=0;j<NT*dofs;j++) {

         Rs[i]   += ( (w2_t*DM2[i][j] + KHB[i][j])*eta[j] );
         num1[i] += (KHB[i][j] * eta[j]); 
         num2[i] += (DM2[i][j] * eta[j]);

  }


  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  OmegaHBcheck = -num/den;
//w2_t = OmegaHBcheck;

  num = 0.0;
  den = 0.0;
  for (int i=0;i<NT*dofs;i++) {
          
         num1[i] = 0.0;
         num2[i] = -cons*Force[i];

  for (int j=0;j<NT*dofs;j++) {

         num1[i] += (w2_t * DM2[i][j] * eta[j] + KHB[i][j] * eta[j]);
         
  }

////num += (Force[i] * num1[i]);
////den += (Force[i] * Force[i]); 
  num += (num2[i] * num1[i]);
  den += (num2[i] * num2[i]); 

  } 

  FS2 = -num/den;

  vf_new = sqrt(FS2);
  w_t    = w_a*sqrt(OmegaHBcheck);

  OmegaHBnew = OmegaHB + 0.3*(w_t - OmegaHB);
  vf         = vf_old + 0.3*(vf_new - vf_old);
  //OmegaHBnew   = w_t;
  //vf           = 2.0*vs/sqrt(mu);

  PeriodHBnew  = 2*PI_NUMBER/OmegaHBnew;
  FlutterSpeed = vf;
  error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;
  error_value_vf  = (vf_new - vf_old)/vf_old;
//cout <<  "OmegaHB= " << OmegaHBnew << endl;  
//cout <<  "Vf= " << vf << endl;  

  }
//su2double NR_error, zeta, dzeta, vf_new;
//su2double gamma = 0.0000001;


//for (int IJK=0;IJK<10;IJK++) {
//L2norm = 0.0;
//num = 0.0;
//den = 0.0;
//for (int i=0;i<NT*dofs;i++) {
//        
//       num2[i] = -cons*Force[i];
//       Rs[i]   = -cons*vf2*Force[i];
//       num1[i] = 0.0;

//for (int j=0;j<NT*dofs;j++) {

//       Rs[i]   += ((OmegaHBcheck * DM2[i][j] + KHB[i][j])*eta[j]);
//       num1[i] += (OmegaHBcheck * DM2[i][j] * eta[j] + KHB[i][j] * eta[j]); 
//       //num2[i] += (DM2[i][j] * eta[j]);

//}

//num += (num1[i] * num2[i]);
//den += (num2[i] * num2[i]); 

//L2norm += (0.5*(Rs[i]*Rs[i])); 

//} 

////gamma = L2norm;

//zeta  = den*vf2*vf2*vf + num*vf2*vf - gamma;
//dzeta = 5*den*vf2*vf + 3*num*vf2;

//vf_new = vf_old - zeta/dzeta;

//vf2 = -num/den;
//vf  = sqrt(vf2);
//NR_error = (vf_new - vf_old)/vf_old;

//}
//cout << "N-R error = " << NR_error << endl;
//
//error_value = (vf_new-vf)/vf;
//vf = vf_new;

//w_t = w_a*sqrt(OmegaHBcheck);

//OmegaHBnew   = w_t;
////vf           = 2.0*vs/sqrt(mu);

//PeriodHBnew  = 2*PI_NUMBER/OmegaHBnew;
//FlutterSpeed = vf;
//error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;
//cout <<  "OmegaHB= " << OmegaHBnew << endl;  
//cout <<  "Vf= " << vf << endl;  


//su2double num11=0.0, num12=0.0, num21=0.0, num22=0.0, den11=0.0, den22=0.0;
//cons = vf*vf/PI_NUMBER;
//L2norm = 0.0;
//num = 0.0;
//den = 0.0;

//for (int i=0;i<NT*dofs;i++) {
//        
//       num1[i] = 0.0;
//       Rs[i]   = -cons*Force[i];
//       num2[i] = 0.0;

//for (int j=0;j<NT*dofs;j++) {

//       Rs[i]   += ((OmegaHB * OmegaHB * DM2[i][j]/w_a/w_a + KHB[i][j])*eta[j]);
//       num2[i] += (KHB[i][j] * eta[j]); 
//       num1[i] += (DM2[i][j] * eta[j]);

//}

//num11 += (num1[i] * num1[i]);
//num21 += (num1[i] * Force[i]); 

//num12 += (num1[i] * Force[i]);
//num22 += (Force[i] * Force[i]);  

//den11 += (num1[i]*num2[i]);
//den22 += (Force[i]*num2[i]);

//L2norm += (0.5*(Rs[i]*Rs[i])); 

//}

//vector<su2double> OmegaSol(2,0.0);
//vector<vector<su2double> > AS(2,vector<su2double>(3,0.0));

//AS[0][0] =  num11;
//AS[1][0] = -num21;
//AS[0][1] = -num12/PI_NUMBER;
//AS[1][1] =  num22/PI_NUMBER;

//AS[0][2] = -den11;
//AS[1][2] =  den22;

//Gauss_Elimination(AS, OmegaSol);

//OmegaHBnew   = w_a*sqrt(OmegaSol[0]);
//FlutterSpeed = sqrt(OmegaSol[1]);
//PeriodHBnew  = 2*PI_NUMBER/OmegaHBnew;
//error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;
//
////////////////////////////////////////////////////////////////////
//
//
  cout << "New HB Omega = " << (OmegaHBnew) << " / L2NORM = " << L2norm << " / Vf = " << FlutterSpeed << endl;
  cout << "Error in omega calc = " << error_value << endl;
  cout << "Error in VF calc = " << error_value_vf << endl;
 
  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::Frequency_Update_2D2(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  vector<su2double> clh(NT,0);
  vector<su2double> cdh(NT,0); 
  vector<su2double> cmh(NT,0);
 
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double mu       = config->GetAeroelastic_Airfoil_Mass_Ratio();
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
  su2double Uinf = config->GetVelocity_FreeStream()[0];
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency (new)" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  HB_Operator(config, DD, EE, NT, OmegaHB);

//


  vector<vector<su2double>> KHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> MHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  MHB[i*dofs+j][i*dofs+k] = M[j][k];
	  }
	  }
	
  }  
  vector<su2double> Force(4,0);
  vector<su2double> eta(4,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> qh(2,vector<su2double>(NT,0.0));
  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
  }

 for (int i=0;i<NT;i++){  
 for (int j=0;j<NT;j++){  
    qh[0][i] += (EE[i][j]*q[0][j]);
    qh[1][i] += (EE[i][j]*q[1][j]);
 } 
 }

          eta[0] = qh[0][2];
          eta[1] = qh[1][2];
          eta[2] = qh[0][1];
          eta[3] = qh[1][1];

  cout << "ETA" << endl;
  for (int i=0;i<4;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

  }

  for (int i=0;i<NT;i++){  
  for (int j=0;j<NT;j++){  
    clh[i] += (EE[i][j]*cl[j]);
    cmh[i] += (EE[i][j]*cm[j]);
  } 
  }

          Force[0] = -clh[2];
          Force[1] = -2*cmh[2];
          Force[2] = -clh[1];
          Force[3] = -2*cmh[1];

  cout << "LIFT    /    MOMENT" << endl;
  for(int count=0;count<NT;count++){

               cout  << cl[count] << " " << cm[count] << "\n" << endl; 

  }
//
//////////////////////////////////////////////////////////////////////////
//
  su2double FlutterSpeed = vf;
  su2double L2norm;

  su2double vf_t = vf*sqrt(mu)/2.0;
  su2double Omega_t = OmegaHB/Uinf;

  su2double OmegaHBnew, OmegaCheck;
  su2double PeriodHBnew;
  su2double error_value;

  su2double cons = 4.0/PI_NUMBER/mu;

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<4;i++) {
          
         num1[i] = -cons*Force[i];
         Rs[i]   = -cons*Force[i];
         num2[i] = 0.0;

  for (int j=0;j<4;j++) {

         Rs[i]   += ( (-Omega_t * Omega_t * MHB[i][j] + KHB[i][j]/vf_t/vf_t) *eta[j] );
         num1[i] += (KHB[i][j] * eta[j]/vf_t/vf_t); 
         num2[i] += (MHB[i][j] * eta[j]);

  }


  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

  OmegaCheck  = num/den;
  Omega_t     = sqrt(OmegaCheck);
  OmegaHBnew  = Uinf*Omega_t;
  PeriodHBnew = 2*PI_NUMBER/OmegaHBnew;
  error_value = (OmegaHBnew - OmegaHB)/OmegaHB;
  cout <<  "OmegaHB= " << OmegaHBnew << endl;  


///////////////////////////////////////////////////////////////////////////////////////////  
  
  cout << "New HB Omega = " << (OmegaHBnew) << " / L2NORM = " << L2norm << " / Vf = " << FlutterSpeed << endl;
  cout << "Error in omega calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);

  config->SetStr_L2_norm(L2norm);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::Frequency_Velocity_Update_2D2(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  vector<su2double> clh(NT,0);
  vector<su2double> cdh(NT,0); 
  vector<su2double> cmh(NT,0);
 
  su2double num = 0.0;
  su2double den = 0.0;
  su2double lum = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double mu       = config->GetAeroelastic_Airfoil_Mass_Ratio();
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
  su2double Uinf = config->GetVelocity_FreeStream()[0];
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency & Velocity (new)" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  HB_Operator(config, DD, EE, NT, OmegaHB);

//


  vector<vector<su2double>> KHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> MHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  MHB[i*dofs+j][i*dofs+k] = M[j][k];
	  }
	  }
	
  }  
  vector<su2double> Force(4,0);
  vector<su2double> eta(4,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> qh(2,vector<su2double>(NT,0.0));
  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
  }

 for (int i=0;i<NT;i++){  
 for (int j=0;j<NT;j++){  
    qh[0][i] += (EE[i][j]*q[0][j]);
    qh[1][i] += (EE[i][j]*q[1][j]);
 } 
 }

          eta[0] = qh[0][2];
          eta[1] = qh[1][2];
          eta[2] = qh[0][1];
          eta[3] = qh[1][1];

  cout << "ETA" << endl;
  for (int i=0;i<4;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

  }

  for (int i=0;i<NT;i++){  
  for (int j=0;j<NT;j++){  
    clh[i] += (EE[i][j]*cl[j]);
    cmh[i] += (EE[i][j]*cm[j]);
  } 
  }

          Force[0] = -clh[2];
          Force[1] = -2*cmh[2];
          Force[2] = -clh[1];
          Force[3] = -2*cmh[1];

  cout << "LIFT    /    MOMENT" << endl;
  for(int count=0;count<NT;count++){

               cout  << cl[count] << " " << cm[count] << "\n" << endl; 

  }
//
//////////////////////////////////////////////////////////////////////////
//
  su2double FlutterSpeed;
  su2double FS2;
  su2double L2norm;

  su2double vf_t = vf*sqrt(mu)/2.0;
  su2double Omega_t = OmegaHB/Uinf;
//  su2double vf_t = vf;
//  su2double Omega_t = OmegaHB/w_a;

  su2double OmegaHBnew, OmegaCheck = Omega_t*Omega_t, vf2_t = vf_t*vf_t;
  su2double PeriodHBnew;
  su2double error_value;

  su2double cons = 4.0/PI_NUMBER/mu;
  //su2double cons = 1.0/PI_NUMBER;

  for (int IKL=0;IKL<1;IKL++) {

  L2norm = 0.0;
  num = 0.0;
  den = 0.0;
  for (int i=0;i<4;i++) {
          
         num1[i] = -cons*Force[i];
         Rs[i]   = -cons*Force[i];
         num2[i] = 0.0;

  for (int j=0;j<4;j++) {

      Rs[i]   += ( (-Omega_t * Omega_t * MHB[i][j] + KHB[i][j]/vf2_t) *eta[j] );
         //Rs[i]   += ( (-OmegaCheck * vf2_t * MHB[i][j] + KHB[i][j]) *eta[j] );
         num1[i] += (KHB[i][j] * eta[j]/vf2_t); 
         num2[i] += (MHB[i][j] * eta[j]);

  }


  num += (num1[i] * num2[i]);
  den += (num2[i] * num2[i]); 

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 

//  cout << "NUM= " << num << " DEN= " << den << endl;
  OmegaCheck  = num/den;
  Omega_t     = sqrt(OmegaCheck);
  //Omega_t     = w_a*sqrt(OmegaCheck);

  num = 0.0;
  den = 0.0;
  for (int i=0;i<4;i++) {
          
         num2[i] = -cons*Force[i];
	 num1[i] = 0.0;

  for (int j=0;j<4;j++) {

     //    num1[i] += ( -OmegaCheck * MHB[i][j] * eta[j]); 
         //num1[i] += ( -OmegaCheck * MHB[i][j] * eta[j] +  KHB[i][j] * eta[j]);
         num1[i] += ( KHB[i][j] * eta[j] );
	 num2[i] += ( -OmegaCheck * MHB[i][j] * eta[j] );
  }

  num += (num1[i] * num2[i]);
  den += (num1[i] * num1[i]); 

  } 
  //cout << "NUM= " << num << " DEN= " << den << endl;
  FS2  = -num/den;
  
  vf_t = 1.0/sqrt(FS2);
//  vf_t = sqrt(FS2);
  vf   = 2.0*vf_t/sqrt(mu);

  OmegaHBnew  = Omega_t*Uinf;
//
//  OmegaHBnew  = Omega_t;
  PeriodHBnew = 2*PI_NUMBER/OmegaHBnew;
  error_value = (OmegaHBnew - OmegaHB)/OmegaHB;
  cout <<  "OmegaHB= " << OmegaHBnew << endl;  
  cout <<  "Vf= " << vf  << endl;  
  FlutterSpeed = vf;

  }

///////////////////////////////////////////////////////////////////////////////////////////  
  
  cout << "New HB Omega = " << (OmegaHBnew) << " / L2NORM = " << L2norm << " / Vf = " << FlutterSpeed << endl;
  cout << "Error in omega calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::Frequency_Velocity_Update_2D4(CConfig *config, su2double* lift, su2double* drag, su2double* moment, int harmonics, bool &error_flag) {

  int NT = harmonics, NH = (NT-1)/2, dofs = 2;
  vector<su2double> num1(NT*dofs,0);
  vector<su2double> num2(NT*dofs,0);
  vector<su2double> Rs(NT*dofs,0);
  
  vector<su2double> cl(NT,0);
  vector<su2double> cd(NT,0); 
  vector<su2double> cm(NT,0);
  vector<su2double> clh(NT,0);
  vector<su2double> cdh(NT,0); 
  vector<su2double> cmh(NT,0);
 
  su2double num = 0.0;
  su2double den = 0.0;
   /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double mu       = config->GetAeroelastic_Airfoil_Mass_Ratio();
  su2double wr  = w_h/w_a;
  su2double w_alpha = w_a;
  su2double Uinf = config->GetVelocity_FreeStream()[0];
//  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Alpha   = config->GetAoA()*PI_NUMBER/180.0;
  unsigned long TimeIter = config->GetTimeIter();
 
  vector<su2double> xi(2,0.0);

//
  cout << "Updating Frequency & Velocity (new)" << endl;
  //
 // vector<vector<su2double>> Kdofs(dofs, vector<su2double>(dofs,0.0));
// Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

  /*--- Get simualation period from config file ---*/
  su2double Period = config->GetHarmonicBalance_Period();

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();

  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double OmegaHB_old = config->GetHB_frequency_old();

  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0));
  HB_Operator(config, DD, EE, NT, OmegaHB);

//


  vector<vector<su2double>> KHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }  
  vector<vector<su2double>> MHB(4,vector<su2double>(4,0.0));
  for(int i=0;i<2;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  MHB[i*dofs+j][i*dofs+k] = M[j][k];
	  }
	  }
	
  }  
  vector<su2double> Force(4,0);
  vector<su2double> eta(4,0.0);

  vector<vector<su2double>> q(2,vector<su2double>(NT,0.0));
  vector<vector<su2double>> qh(2,vector<su2double>(NT,0.0));
  for (int i=0;i<NT;i++){  
    q[1][i] = config->GetHB_pitch(i);
    q[0][i] = config->GetHB_plunge(i);
  }

 for (int i=0;i<NT;i++){  
 for (int j=0;j<NT;j++){  
    qh[0][i] += (EE[i][j]*q[0][j]);
    qh[1][i] += (EE[i][j]*q[1][j]);
 } 
 }

          eta[0] = qh[0][2];
          eta[1] = qh[1][2];
          eta[2] = qh[0][1];
          eta[3] = qh[1][1];

  cout << "ETA" << endl;
  for (int i=0;i<4;i++) {
          cout << eta[i] << endl;
  }
  cout << endl;

  for(int i=0;i<harmonics;i++) {

          cl[i] = lift[i]*cos(Alpha) + drag[i]*sin(Alpha);
          cd[i] = -lift[i]*sin(Alpha) + drag[i]*cos(Alpha);
        
          cm[i] = moment[i];

  }

  for (int i=0;i<NT;i++){  
  for (int j=0;j<NT;j++){  
    clh[i] += (EE[i][j]*cl[j]);
    cmh[i] += (EE[i][j]*cm[j]);
  } 
  }

          Force[0] = -clh[2];
          Force[1] = -2*cmh[2];
          Force[2] = -clh[1];
          Force[3] = -2*cmh[1];

  cout << "LIFT    /    MOMENT" << endl;
  for(int count=0;count<NT;count++){

               cout  << cl[count] << " " << cm[count] << "\n" << endl; 

  }
//
//////////////////////////////////////////////////////////////////////////
//
  su2double L2norm;

//  su2double vf_t = vf*sqrt(mu)/2.0;

  su2double vf_t = vf;
  su2double ovs2 = 1.0/vf_t/vf_t;
  su2double ovs3 = 1.0/vf_t/vf_t/vf_t;
  su2double vs2  = vf_t*vf_t;

//  su2double Omega_t  = OmegaHB/Uinf;
  su2double Omega_t  = OmegaHB/w_a;
  su2double Omega2_t = Omega_t * Omega_t;

  su2double OmegaHBnew, Omega_t_new, vf_t_new;
  su2double PeriodHBnew;
  su2double error_value;

  //su2double cons = 4.0/PI_NUMBER/mu;
  su2double cons = 1.0/PI_NUMBER;

//L2norm = 0.0;
//for (int i=0;i<4;i++) {
//        
//       num1[i] = 0.0;
//       num2[i] = 0.0;
//       Rs[i]   = -cons*Force[i];

//for (int j=0;j<4;j++) {

//       Rs[i]   += (-Omega2_t * MHB[i][j] * eta[j] + ovs2 * KHB[i][j]* eta[j]);
//       num2[i] += (-2.0 * ovs3 * KHB[i][j] * eta[j]); 
//       num1[i] += (-2.0 * Omega_t * MHB[i][j] * eta[j]);

//}

//L2norm += (0.5*(Rs[i]*Rs[i])); 

//} 
  L2norm = 0.0;
  for (int i=0;i<4;i++) {
          
         num1[i] = 0.0;
         num2[i] = -2.0*cons*vf_t*Force[i];
         Rs[i]   = -cons*vs2*Force[i];

  for (int j=0;j<4;j++) {

         Rs[i]   += (-Omega2_t * MHB[i][j] * eta[j] + KHB[i][j]* eta[j]);
         //num2[i] += (-2.0 * Omega2_t * vf_t * MHB[i][j] * eta[j]); 
         num1[i] += (-2.0 * Omega_t * MHB[i][j] * eta[j]);

  }

  L2norm += (0.5*(Rs[i]*Rs[i])); 

  } 


  vector<vector<su2double>> Asys(2,vector<su2double>(2,0.0));
  vector<vector<su2double>> Asys_inv(2,vector<su2double>(2,0.0));

  cout << "REAL" << endl;
  cout << "Residual= [" << Rs[0] << ", " << Rs[1] << "]" << endl;
  cout << "Num1 = [" << num1[0] << ", " << num1[1] << "]" << endl;
  cout << "Num2 = [" << num2[0] << ", " << num2[1] << "]" << endl; 
  cout << "IMG" << endl;
  cout << "Residual= [" << Rs[2] << ", " << Rs[3] << "]" << endl;
  cout << "Num1 = [" << num1[2] << ", " << num1[3] << "]" << endl;
  cout << "Num2 = [" << num2[2] << ", " << num2[3] << "]" << endl;

//Asys[0][0] = num1[0];
//Asys[1][0] = num1[1];
//Asys[0][1] = num2[0];
//Asys[1][1] = num2[1];

  Asys[0][0] = num1[1];
  Asys[1][0] = num1[3];
  Asys[0][1] = num2[1];
  Asys[1][1] = num2[3];

//Asys[0][0] = num1[2];
//Asys[1][0] = num1[3];
//Asys[0][1] = num2[2];
//Asys[1][1] = num2[3];

  Inverse_matrix2D(Asys, Asys_inv);

  cout << "Asys_inv" << endl;
  cout << "|" << Asys_inv[0][0] << " " << Asys_inv[0][1] << "|" << endl;
  cout << "|" << Asys_inv[1][0] << " " << Asys_inv[1][1] << "|" << endl;

  su2double zeta = 0.1;

  Omega_t_new = Omega_t - zeta * (Asys_inv[0][0]*Rs[0] + Asys_inv[0][1] * Rs[1]);
  vf_t_new    = vf_t    - zeta * (Asys_inv[1][0]*Rs[0] + Asys_inv[1][1] * Rs[1]);

//Omega_t_new = Omega_t - zeta * (Asys_inv[0][0]*Rs[2] - Asys_inv[0][1] * Rs[2]);
//vf_t_new    = vf_t    - zeta * (Asys_inv[1][0]*Rs[3] - Asys_inv[1][1] * Rs[3]);

//OmegaHBnew  = Uinf*Omega_t_new;
//vf          = 2.0*vf_t_new/sqrt(mu);

  OmegaHBnew  = w_a*Omega_t_new;
  vf          = vf_t_new;

  PeriodHBnew = 2*PI_NUMBER/OmegaHBnew;

  error_value = (OmegaHBnew - OmegaHB)/OmegaHB;

///////////////////////////////////////////////////////////////////////////////////////////  
  
  cout << "New HB Omega = " << (OmegaHBnew) << " / L2NORM = " << L2norm << " / Vf = " << vf << endl;
  cout << "Error in omega calc = " << error_value << endl;

  //config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(abs(L2norm) < 0.00000000000001)) {error_flag = true;} 

  config->SetHarmonicBalance_Period(PeriodHBnew);

  config->SetStr_L2_norm(L2norm);

  config->SetAeroelastic_Flutter_Speed_Index(vf);

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

}

void CSolver::SolveWing_HB_Thomas_Flutter(CGeometry *geometry, CConfig *config, unsigned short iMarker, vector<vector<su2double>>& displacements, vector<su2double> cl, vector<su2double> cd, vector<su2double> cm, int harmonics) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/

  /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha  = w_a;
  su2double mu       = config->GetAeroelastic_Airfoil_Mass_Ratio();
  su2double vf       = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b        = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  unsigned long TimeIter = config->GetTimeIter();
  const su2double DEG2RAD = PI_NUMBER/180.0; 
  su2double dtau;
  su2double PitchAmpl = config->GetPitching_Ampl(2)*DEG2RAD;

  cout << "POIUUDJFWKFWEKF" << endl;
  cout << "xa= " << x_a << ", r2a= " << r_a*r_a << ", wh/wa= " << wr << ", Vf= " << vf << ", b= " << b << endl;

   int dofs = 2, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(2,0.0);

  su2double Uinf = config->GetVelocity_FreeStream()[0];
  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  su2double vs      = vf*sqrt(mu)/2.0;
  su2double omega_t = OmegaHB/w_a/vs; 
  //su2double vs      = vf;
 // su2double omega_t = OmegaHB/w_a;
 
 /*--- Get simualation period from config file ---*/
  su2double Period = 2*PI_NUMBER/OmegaHB;

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();
//
////////////////////////////////////////////////////////////////////////////////
/*--- Eigenvectors and Eigenvalues of the Generalized EigenValue Problem. ---*/
//
  // Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

//
  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0)); 
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  

  HB_Operator(config, DD, EE, NT, OmegaHB);

//  
  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT-1;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }

  vector<vector<su2double>> MHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT-1;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  MHB[i*dofs+j][i*dofs+k] = M[j][k];
	  }
	  }
	
  }
 
  /*--- Forcing Term ---*/
//
/////////////////////////////////////////////////////////
//        
  cout << "LIFT   /   MOMENT   "  << endl;
  for(int i=0;i<NT;i++) { 

   cout << cl[i] << " " << cm[i] << endl;
        
  }

  vector<su2double> Force(4,0);
  vector<su2double> clh(NT,0.0);
  vector<su2double> cmh(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       clh[i] += EE[i][j]*cl[j];
       cmh[i] += EE[i][j]*cm[j];
        
  }
  }

  su2double CL0  = clh[0], ImCL = clh[1], ReCL = clh[2];
  su2double CM0  = cmh[0], ImCM = cmh[1], ReCM = cmh[2];

  Force[0] = -ReCL;
  Force[1] = -2*ReCM;
  Force[2] = -ImCL;
  Force[3] = -2*ImCM;

  cout << "FORCE" << endl;    
  for(int j=0;j<4;j++) { 

	  cout << Force[j] << "\n";
    
  }  
  cout << endl;
//
//////////////////////////////////////////////////////////
//
  vector<su2double> deta(4,0.0);
  vector<su2double> eta_old(4,0.0);
  vector<su2double> eta_new(4,0.0);  
  vector<su2double> v_vec(4,0.0);  
  vector<su2double> h(NT,0.0); 
  vector<su2double> alpha(NT,0.0);

  for (int i=0;i<NT;i++){  
    h[i]     = config->GetHB_plunge(i);
    alpha[i] = config->GetHB_pitch(i);
  }
   
  cout << "OLD: PLUNGE  -  PITCH" << endl;    
  for(int j=0;j<NT;j++) { 

	  cout << h[j] << "  "<< alpha[j] << "\n";
    
  }  
  cout << endl;
 
  vector<su2double> hh(NT,0.0);
  vector<su2double> ah(NT,0.0);

  cout << "alpha_hat = " << endl;
  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       hh[i] += EE[i][j]*h[j];
       ah[i] += EE[i][j]*alpha[j];
  }
       cout << ah[i] << endl; 
  }

  su2double H0  = hh[0], ImH = hh[1], ReH = hh[2];
  su2double A0  = ah[0], ImA = ah[1], ReA = ah[2];

  ImA = 0.0;
  ReA = PitchAmpl;

  v_vec[0] = ReH;
  v_vec[1] = ReA;
  v_vec[2] = ImH;
  v_vec[3] = ImA;

          eta_old[0] = v_vec[0];
          eta_old[1] = vs;
          eta_old[2] = v_vec[2];
	  eta_old[3] = omega_t;

    cout << "eta_old" << endl;
    for(int j=0;j<4;j++) { 
             cout << eta_old[j] << "\n";
    }
    cout << endl;

//
/////////////////////////////////////////////////////////
//
  vector<su2double> RS(4,0.0);
  vector<vector<su2double> > ASys(4,vector<su2double>(5,0.0));
  vector<vector<su2double> > JR(4,vector<su2double>(4,0.0));
  su2double L2norm=0.0;
  //su2double vs      = Uinf/w_a;
  //su2double omega_t = OmegaHB/Uinf;

  su2double omega2_t = omega_t*omega_t, ovs=1.0/vs, ovs2 = 1.0/(vs*vs), vs2 = vs*vs, ovs3 = 1.0/(vs*vs*vs), error_tol;

  int NRiter= 15;
  vector<su2double> lambda(4,1);
  lambda[1] = 0.1;
  lambda[3] = 0.1; 
       
  for(int IJK=0;IJK<NRiter;IJK++) {

	  cout << IJK << endl;
	  for(int i=0;i<4;i++) {

                  JR[i][3] = 0.0;
                  for(int j=0;j<4;j++){

                	  JR[i][3] += (-2.0*omega_t*MHB[i][j]*v_vec[j]*mu); 

                  }
          }

          for(int i=0;i<4;i++) {

                  //JR[i][0] = -8.0*vs*Force[i]/PI_NUMBER;
		  JR[i][1] = 0.0;
                  for(int j=0;j<4;j++){

                	  //JR[i][0] += (-2.0*vs*omega2_t*MHB[i][j]*v_vec[j]*mu); 
			  JR[i][1] += ( -2.0*ovs3*KHB[i][j]*v_vec[j]*mu);

                  }
          }
          
	  JR[0][0] = -omega2_t*MHB[0][0]*mu + KHB[0][0]*ovs2*mu;
	  JR[1][0] = -omega2_t*MHB[1][0]*mu + KHB[1][0]*ovs2*mu;
	  JR[2][0] = -omega2_t*MHB[2][0]*mu + KHB[2][0]*ovs2*mu;
	  JR[3][0] = -omega2_t*MHB[3][0]*mu + KHB[3][0]*ovs2*mu;
                                                           
	  JR[0][2] = -omega2_t*MHB[0][2]*mu + KHB[0][2]*ovs2*mu;
	  JR[1][2] = -omega2_t*MHB[1][2]*mu + KHB[1][2]*ovs2*mu;
	  JR[2][2] = -omega2_t*MHB[2][2]*mu + KHB[2][2]*ovs2*mu;
	  JR[3][2] = -omega2_t*MHB[3][2]*mu + KHB[3][2]*ovs2*mu;

	  cout << "JACOBIAN" << endl; 
	  for(int i=0;i<4;i++) {
          for(int j=0;j<4;j++) {  

		  cout << JR[i][j] << " "; 
	  
	  }
	  cout << endl;
	  }
	  //


	  cout << "RESID" << endl;
	  for(int i=0;i<4;i++) {

		  RS[i] = -4.0*Force[i]/PI_NUMBER;
		  //RS[i] = -Force[i]/PI_NUMBER;
		  for(int j=0;j<4;j++){

			  RS[i] += ((-omega2_t*MHB[i][j] + KHB[i][j]*ovs2)*v_vec[j]*mu); 
		//
//			  RS[i] += ( (-omega2_t*MHB[i][j]*vs2 + KHB[i][j])*v_vec[j]*mu ); 

		  }

			  cout << RS[i] << endl;
	  }

	  for(int i=0;i<4;i++) {
          for(int j=0;j<4;j++) {  

		  ASys[i][j] = JR[i][j];

	  }
	  }
	  for(int i=0;i<4;i++) {

		  ASys[i][4] = -RS[i];

	  }

	  for(int i=0;i<4;i++) {

		  deta[i] = 0.0;

	  }

	  ////////////////////////////////
	  Gauss_Elimination(ASys, deta);//
	  ////////////////////////////////

	  cout << "D_ETA    /   RS" << endl;
	  error_tol = 0.0;
	  for(int i=0;i<4;i++) {

		  cout << deta[i] << " " << RS[i] << endl;

	  eta_new[i] = eta_old[i] + lambda[i]*deta[i];

	  error_tol += (deta[i]*deta[i]); 

	  eta_old[i] = eta_new[i];

	  L2norm += (RS[i]*RS[i]);

	  }
	  error_tol = sqrt(error_tol);

	  cout << "Error_NR= " << error_tol << endl;

	  if (error_tol > 100) break;

	  v_vec[0] = eta_new[0];
	  vs       = eta_new[1];
          v_vec[2] = eta_new[2];
	  omega_t  = eta_new[3];
	  
	  omega2_t = omega_t*omega_t;
	  ovs2     = 1.0/vs/vs;
	  ovs      = 1.0/vs;
	  vs2      = vs*vs;
	  ovs3     = 1.0/vs/vs/vs;
  }

          cout << "eta_new" << endl;
	  for(int j=0;j<4;j++) { 
             cout << eta_new[j] << "\n";
    
	  }
	  cout << endl;

///////// if (error_tol>10) {

/////////  vs      = vf*sqrt(mu)/2.0;
/////////  omega_t = OmegaHB/w_a/vs;
//
/////////  v_vec[0] = (ReH); 
/////////  v_vec[1] = PitchAmpl/2.0;
/////////  v_vec[2] = (ImH);
/////////  v_vec[3] = 0.0;

///////// }
//
/////////////////////////////////////////////////////////////
//
 
  su2double OmegaHBnew   = omega_t*w_a*vs;
  su2double FlutterSpeed = 2.0*vs/sqrt(mu); 
  //su2double OmegaHBnew   = omega_t*w_a;
  //su2double FlutterSpeed = vs; 
  su2double PeriodHBnew  = 2.0*PI_NUMBER/OmegaHBnew;
  su2double error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;

  config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
 cout << "Omega HB= " << OmegaHBnew << " | FlutterSpeed= " << FlutterSpeed << endl; 
//  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(error_tol < 0.00000000000001)) {error_flag = true;} 

  cout << "Freq Error= " << error_value << " / NR error= " << error_tol << endl;
  config->SetHarmonicBalance_Period(PeriodHBnew);
  config->SetStr_L2_norm(sqrt(L2norm));

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

//
/////////////////////////////////////////////////////////////
//
 
  //ReA = PitchAmpl;
  //ImA = 0.0;
  ReH = v_vec[0];
  ImH = v_vec[2];

  hh[0] = 0.0; hh[1] = ImH; hh[2] = ReH;
  ah[0] = 0.0; ah[1] = ImA; ah[2] = ReA;

	    cout << "TIMEHB" << endl;
	    vector<su2double> time_inst(NT,0.0);	
	    for(int i=0;i<NT;i++) {

                    time_inst[i]  = i*PeriodHBnew/NT;
                    cout << time_inst[i] << "\n";
	    }
	    cout << endl;

   	    for(int i=0;i<NT;i++) {

		    h[i]     = hh[0];
	            alpha[i] = ah[0];  

		    for(int j=0;j<NH;j++) {

		    h[i]     = h[i] + hh[2*j+1]*cos((j+1)*OmegaHBnew*time_inst[i]) 
			            + hh[2*j+2]*sin((j+1)*OmegaHBnew*time_inst[i]);
	            alpha[i] = alpha[i] + ah[2*j+1]*cos((j+1)*OmegaHBnew*time_inst[i])
				        + ah[2*j+2]*sin((j+1)*OmegaHBnew*time_inst[i]);

		    }

	    }

  vector<su2double> alpha_d(NT,0.0);
  vector<su2double> h_d(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       h_d[i] += (OmegaHBnew*DD[i][j]*h[j]);
       alpha_d[i] += (OmegaHBnew*DD[i][j]*alpha[j]);
        
  }
  }

  /*--- Set the solution of the structural equations ---*/
  for (int i=0;i<NT;i++) {

  displacements[i][0] = b*h[i];
  displacements[i][1] = alpha[i];
  displacements[i][2] = b*h_d[i];
  displacements[i][3] = alpha_d[i];

  }
  for (int i=0;i<NT;i++){  
    config->SetHB_pitch(displacements[i][1], i);
    config->SetHB_plunge(displacements[i][0]/b, i);
    config->SetHB_pitch_rate(displacements[i][3], i);
    config->SetHB_plunge_rate(displacements[i][2], i);
  }

//
/////////////////////////////////////////////////////////////
//
  
   cout << "NEW: PLUNGE  -  PITCH" << endl;
      for(int j=0;j<NT;j++) { 
             cout << h[j] << "  "<< alpha[j] << "\n";
    }
      cout << endl;
 
}

void CSolver::SolveWing_HB_Thomas_Velocity(CGeometry *geometry, CConfig *config, unsigned short iMarker, vector<vector<su2double>>& displacements, vector<su2double> cl, vector<su2double> cd, vector<su2double> cm, int harmonics) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/

  /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double wr  = w_h/w_a;
  su2double w_alpha  = w_a;
  su2double mu       = config->GetAeroelastic_Airfoil_Mass_Ratio();
  su2double vf       = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b        = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  unsigned long TimeIter = config->GetTimeIter();
  const su2double DEG2RAD = PI_NUMBER/180.0; 
  su2double dtau;
  su2double PitchAmpl = config->GetPitching_Ampl(2)*DEG2RAD;

  cout << "xa= " << x_a << ", r2a= " << r_a*r_a << ", wh/wa= " << wr << ", Vf= " << vf << ", b= " << b << endl;

   int dofs = 2, NT = harmonics, NH = (NT-1)/2;
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(2,0.0);

  su2double Uinf = config->GetVelocity_FreeStream()[0];
  //su2double OmegaHB = 2 * PI_NUMBER / Period;
  su2double OmegaHB = config->GetOmega_HB()[1];
  //su2double vs      = vf*sqrt(mu)/2.0;
  //su2double omega_t = OmegaHB/w_a/vs; 
  su2double vs      = vf;
  su2double omega_t = OmegaHB/w_a;
 
 /*--- Get simualation period from config file ---*/
  su2double Period = 2*PI_NUMBER/OmegaHB;

  /*--- Non-dimensionalize the input period, if necessary.      */
  Period /= config->GetTime_Ref();
//
////////////////////////////////////////////////////////////////////////////////
/*--- Eigenvectors and Eigenvalues of the Generalized EigenValue Problem. ---*/
//
  // Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > M_inv(2,vector<su2double>(2,0.0)); 
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  Inverse_matrix2D(M, M_inv);

  // Stiffness Matrix
  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  K[0][0] = (w_h/w_a)*(w_h/w_a);
  K[0][1] = 0.0;
  K[1][0] = 0.0;
  K[1][1] = r_a*r_a;

   // Stiffness Matrix
  vector<vector<su2double> > T(2,vector<su2double>(2,0.0));
  T[0][0] = 2*xi[0]*(w_h/w_a)*(w_h/w_a);
  T[0][1] = 0.0;
  T[1][0] = 0.0;
  T[1][1] = 2*xi[1]*r_a*r_a;

//
  vector<vector<su2double>> DD(NT, vector<su2double>(NT,0.0));
  vector<vector<su2double>> EE(NT, vector<su2double>(NT,0.0)); 
  vector<vector<su2double>> DHB(NT*dofs, vector<su2double>(NT*dofs,0.0));  

  HB_Operator(config, DD, EE, NT, OmegaHB);

//  
  vector<vector<su2double>> KHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT-1;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  KHB[i*dofs+j][i*dofs+k] = K[j][k];
	  }
	  }
	
  }

  vector<vector<su2double>> MHB(NT*dofs,vector<su2double>(NT*dofs,0.0));
  for(int i=0;i<NT-1;i++) {

	  for (int j=0;j<dofs;j++){
          for (int k=0;k<dofs;k++){
	  MHB[i*dofs+j][i*dofs+k] = M[j][k];
	  }
	  }
	
  }
 
  /*--- Forcing Term ---*/
//
/////////////////////////////////////////////////////////
//        
  cout << "LIFT   /   MOMENT   "  << endl;
  for(int i=0;i<NT;i++) { 

   cout << cl[i] << " " << cm[i] << endl;
        
  }

  vector<su2double> Force(4,0);
  vector<su2double> clh(NT,0.0);
  vector<su2double> cmh(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       clh[i] += EE[i][j]*cl[j];
       cmh[i] += EE[i][j]*cm[j];
        
  }
  }

  su2double CL0  = clh[0], ImCL = clh[1], ReCL = clh[2];
  su2double CM0  = cmh[0], ImCM = cmh[1], ReCM = cmh[2];

  Force[0] = -ReCL;
  Force[1] = -2*ReCM;
  Force[2] = -ImCL;
  Force[3] = -2*ImCM;

  cout << "FORCE" << endl;    
  for(int j=0;j<4;j++) { 

	  cout << Force[j] << "\n";
    
  }  
  cout << endl;
//
//////////////////////////////////////////////////////////
//
  vector<su2double> deta(4,0.0);
  vector<su2double> eta_old(4,0.0);
  vector<su2double> eta_new(4,0.0);  
  vector<su2double> v_vec(4,0.0);  
  vector<su2double> h(NT,0.0); 
  vector<su2double> alpha(NT,0.0);

  for (int i=0;i<NT;i++){  
    h[i]     = config->GetHB_plunge(i);
    alpha[i] = config->GetHB_pitch(i);
  }
   
  cout << "OLD: PLUNGE  -  PITCH" << endl;    
  for(int j=0;j<NT;j++) { 

	  cout << h[j] << "  "<< alpha[j] << "\n";
    
  }  
  cout << endl;
 
  vector<su2double> hh(NT,0.0);
  vector<su2double> ah(NT,0.0);

  cout << "alpha_hat = " << endl;
  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       hh[i] += EE[i][j]*h[j];
       ah[i] += EE[i][j]*alpha[j];
  }
       cout << ah[i] << endl; 
  }

  su2double H0  = hh[0], ImH = hh[1], ReH = hh[2];
  su2double A0  = ah[0], ImA = ah[1], ReA = ah[2];

  ImA = 0.0;
  ReA = PitchAmpl;

  v_vec[0] = ReH;
  v_vec[1] = ReA;
  v_vec[2] = ImH;
  v_vec[3] = ImA;

          eta_old[0] = v_vec[0];
          eta_old[1] = v_vec[1];
	  eta_old[2] = v_vec[2];;
          eta_old[3] = vs;

    cout << "eta_old" << endl;
    for(int j=0;j<4;j++) { 
             cout << eta_old[j] << "\n";
    }
    cout << endl;

//
/////////////////////////////////////////////////////////
//
  vector<su2double> RS(4,0.0);
  vector<vector<su2double> > ASys(4,vector<su2double>(5,0.0));
  vector<vector<su2double> > JR(4,vector<su2double>(4,0.0));
  su2double L2norm=0.0;
  //su2double vs      = Uinf/w_a;
  //su2double omega_t = OmegaHB/Uinf;

  su2double omega2_t = omega_t*omega_t, ovs=1.0/vs, ovs2 = 1.0/(vs*vs), vs2 = vs*vs, ovs3 = 1.0/(vs*vs*vs), error_tol;

  int NRiter= 1;
  vector<su2double> lambda(4,0.1);
  //lambda[1] = 0.01;
  //lambda[3] = 0.01; 
       
  for(int IJK=0;IJK<NRiter;IJK++) {

//////////cout << IJK << endl;
//////////for(int i=0;i<4;i++) {

//////////        JR[i][3] = 0.0;
//////////        for(int j=0;j<4;j++){

//////////      	  JR[i][3] += (-2.0*omega_t*MHB[i][j]*v_vec[j]*mu); 

//////////        }
//////////}

          for(int i=0;i<4;i++) {

                  JR[i][3] = -2.0*vs*Force[i]/PI_NUMBER;
		  //JR[i][2] = 0.0;
                  for(int j=0;j<4;j++){

                	  //JR[i][0] += (-2.0*vs*omega2_t*MHB[i][j]*v_vec[j]*mu); 
		//	  JR[i][2] += ( -2.0*ovs3*KHB[i][j]*v_vec[j]*mu);

                  }
          }
          
	  JR[0][0] = -omega2_t*MHB[0][0] + KHB[0][0];
	  JR[1][0] = -omega2_t*MHB[1][0] + KHB[1][0];
	  JR[2][0] = -omega2_t*MHB[2][0] + KHB[2][0];
	  JR[3][0] = -omega2_t*MHB[3][0] + KHB[3][0];
          
          JR[0][1] = -omega2_t*MHB[0][1] + KHB[0][1];
	  JR[1][1] = -omega2_t*MHB[1][1] + KHB[1][1];
	  JR[2][1] = -omega2_t*MHB[2][1] + KHB[2][1];
	  JR[3][1] = -omega2_t*MHB[3][1] + KHB[3][1];
                                                    
	  JR[0][2] = -omega2_t*MHB[0][2] + KHB[0][2];
	  JR[1][2] = -omega2_t*MHB[1][2] + KHB[1][2];
	  JR[2][2] = -omega2_t*MHB[2][2] + KHB[2][2];
	  JR[3][2] = -omega2_t*MHB[3][2] + KHB[3][2];

	  cout << "JACOBIAN" << endl; 
	  for(int i=0;i<4;i++) {
          for(int j=0;j<4;j++) {  

		  cout << JR[i][j] << " "; 
	  
	  }
	  cout << endl;
	  }
	  //


	  cout << "RESID" << endl;
	  for(int i=0;i<4;i++) {

		  RS[i] = -vs2*Force[i]/PI_NUMBER;
		  //RS[i] = -Force[i]/PI_NUMBER;
		  for(int j=0;j<4;j++){

			  RS[i] += ((-omega2_t*MHB[i][j] + KHB[i][j])*v_vec[j]); 
		//
//			  RS[i] += ( (-omega2_t*MHB[i][j]*vs2 + KHB[i][j])*v_vec[j]*mu ); 

		  }

			  cout << RS[i] << endl;
	  }

	  for(int i=0;i<4;i++) {
          for(int j=0;j<4;j++) {  

		  ASys[i][j] = JR[i][j];

	  }
	  }
	  for(int i=0;i<4;i++) {

		  ASys[i][4] = -RS[i];

	  }

	  for(int i=0;i<4;i++) {

		  deta[i] = 0.0;

	  }

	  ////////////////////////////////
	  Gauss_Elimination(ASys, deta);//
	  ////////////////////////////////

	  cout << "D_ETA    /   RS" << endl;
	  error_tol = 0.0;
	  for(int i=0;i<4;i++) {

		  cout << deta[i] << " " << RS[i] << endl;

	  eta_new[i] = eta_old[i] + lambda[i]*deta[i];

	  error_tol += (deta[i]*deta[i]); 

	  eta_old[i] = eta_new[i];

	  L2norm += (RS[i]*RS[i]);

	  }
	  error_tol = sqrt(error_tol);

	  cout << "Error_NR= " << error_tol << endl;

	  if (error_tol > 100) break;

	  v_vec[0] = eta_new[0];
          v_vec[1] = eta_new[1];
	  v_vec[2] = eta_new[2]; 
	  vs       = eta_new[3];

	  ovs2     = 1.0/vs/vs;
	  ovs      = 1.0/vs;
	  vs2      = vs*vs;
	  ovs3     = 1.0/vs/vs/vs;
  }

          cout << "eta_new" << endl;
	  for(int j=0;j<4;j++) { 
             cout << eta_new[j] << "\n";
    
	  }
	  cout << endl;

///////// if (error_tol>10) {

/////////  vs      = vf*sqrt(mu)/2.0;
/////////  omega_t = OmegaHB/w_a/vs;
//
/////////  v_vec[0] = (ReH); 
/////////  v_vec[1] = PitchAmpl/2.0;
/////////  v_vec[2] = (ImH);
/////////  v_vec[3] = 0.0;

///////// }
//
/////////////////////////////////////////////////////////////
//
 
  //su2double OmegaHBnew   = omega_t*w_a*vs;
  //su2double FlutterSpeed = 2.0*vs/sqrt(mu); 
  su2double FlutterSpeed = vs; 
  su2double OmegaHBnew   = OmegaHB;
  //su2double OmegaHBnew   = omega_t*w_a;
  //su2double FlutterSpeed = vs; 
  su2double PeriodHBnew  = 2.0*PI_NUMBER/OmegaHBnew;
  su2double error_value  = (OmegaHBnew - OmegaHB)/OmegaHB;

  config->SetAeroelastic_Flutter_Speed_Index(FlutterSpeed);
 
 cout << "Omega HB= " << OmegaHBnew << " | FlutterSpeed= " << FlutterSpeed << endl; 
//  if ((abs(error_value) < 0.0000001 || OmegaHBnew < 0.001)&&(error_tol < 0.00000000000001)) {error_flag = true;} 

  cout << "Freq Error= " << error_value << " / NR error= " << error_tol << endl;
  config->SetHarmonicBalance_Period(PeriodHBnew);
  config->SetStr_L2_norm(sqrt(L2norm));

  config->SetOmega_HB(0.0, 0);
  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(i*OmegaHBnew, i);

  for (int i=1;i<NH+1;i++)  
    config->SetOmega_HB(-i*OmegaHBnew, NH + i);

//
/////////////////////////////////////////////////////////////
//
 
  //ReA = PitchAmpl;
  //ImA = 0.0;
  ReH = v_vec[0];
  ImH = v_vec[2];
  ReA = v_vec[1];
  ImA = v_vec[3];

  hh[0] = 0.0; hh[1] = ImH; hh[2] = ReH;
  ah[0] = 0.0; ah[1] = ImA; ah[2] = ReA;

	    cout << "TIMEHB" << endl;
	    vector<su2double> time_inst(NT,0.0);	
	    for(int i=0;i<NT;i++) {

                    time_inst[i]  = i*PeriodHBnew/NT;
                    cout << time_inst[i] << "\n";
	    }
	    cout << endl;

   	    for(int i=0;i<NT;i++) {

		    h[i]     = hh[0];
	            alpha[i] = ah[0];  

		    for(int j=0;j<NH;j++) {

		    h[i]     = h[i] + hh[2*j+1]*cos((j+1)*OmegaHBnew*time_inst[i]) 
			            + hh[2*j+2]*sin((j+1)*OmegaHBnew*time_inst[i]);
	            alpha[i] = alpha[i] + ah[2*j+1]*cos((j+1)*OmegaHBnew*time_inst[i])
				        + ah[2*j+2]*sin((j+1)*OmegaHBnew*time_inst[i]);

		    }

	    }

  vector<su2double> alpha_d(NT,0.0);
  vector<su2double> h_d(NT,0.0);

  for(int i=0;i<NT;i++) { 
  for(int j=0;j<NT;j++) { 

       h_d[i] += (OmegaHBnew*DD[i][j]*h[j]);
       alpha_d[i] += (OmegaHBnew*DD[i][j]*alpha[j]);
        
  }
  }

  /*--- Set the solution of the structural equations ---*/
  for (int i=0;i<NT;i++) {

  displacements[i][0] = b*h[i];
  displacements[i][1] = alpha[i];
  displacements[i][2] = b*h_d[i];
  displacements[i][3] = alpha_d[i];

  }
  for (int i=0;i<NT;i++){  
    config->SetHB_pitch(displacements[i][1], i);
    config->SetHB_plunge(displacements[i][0]/b, i);
    config->SetHB_pitch_rate(displacements[i][3], i);
    config->SetHB_plunge_rate(displacements[i][2], i);
  }

//
/////////////////////////////////////////////////////////////
//
  
   cout << "NEW: PLUNGE  -  PITCH" << endl;
      for(int j=0;j<NT;j++) { 
             cout << h[j] << "  "<< alpha[j] << "\n";
    }
      cout << endl;
 
}

void CSolver::SetUpTypicalSectionWingModel(vector<vector<su2double> >& Phi, vector<su2double>& omega, CConfig *config) {

  /*--- Retrieve values from the config file ---*/
  su2double w_h = config->GetAeroelastic_Frequency_Plunge();
  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  su2double x_a = config->GetAeroelastic_CG_Location();
  su2double r_a = sqrt(config->GetAeroelastic_Radius_Gyration_Squared());
  su2double w = w_h/w_a;

  // Mass Matrix
  vector<vector<su2double> > M(2,vector<su2double>(2,0.0));
  M[0][0] = 1;
  M[0][1] = x_a;
  M[1][0] = x_a;
  M[1][1] = r_a*r_a;

  // Stiffness Matrix
  //  vector<vector<su2double> > K(2,vector<su2double>(2,0.0));
  //  K[0][0] = (w_h/w_a)*(w_h/w_a);
  //  K[0][1] = 0.0;
  //  K[1][0] = 0.0;
  //  K[1][1] = r_a*r_a;

  /* Eigenvector and Eigenvalue Matrices of the Generalized EigenValue Problem. */

  vector<vector<su2double> > Omega2(2,vector<su2double>(2,0.0));
  su2double aux; // auxiliary variable
  aux = sqrt(pow(r_a,2)*pow(w,4) - 2*pow(r_a,2)*pow(w,2) + pow(r_a,2) + 4*pow(x_a,2)*pow(w,2));
  Phi[0][0] = (r_a * (r_a - r_a*pow(w,2) + aux)) / (2*x_a*pow(w, 2));
  Phi[0][1] = (r_a * (r_a - r_a*pow(w,2) - aux)) / (2*x_a*pow(w, 2));
  Phi[1][0] = 1.0;
  Phi[1][1] = 1.0;

  Omega2[0][0] = (r_a * (r_a + r_a*pow(w,2) - aux)) / (2*(pow(r_a, 2) - pow(x_a, 2)));
  Omega2[0][1] = 0;
  Omega2[1][0] = 0;
  Omega2[1][1] = (r_a * (r_a + r_a*pow(w,2) + aux)) / (2*(pow(r_a, 2) - pow(x_a, 2)));

  /* Nondimesionalize the Eigenvectors such that Phi'*M*Phi = I and PHI'*K*PHI = Omega */
  // Phi'*M*Phi = D
  // D^(-1/2)*Phi'*M*Phi*D^(-1/2) = D^(-1/2)*D^(1/2)*D^(1/2)*D^(-1/2) = I,  D^(-1/2) = inv(sqrt(D))
  // Phi = Phi*D^(-1/2)

  vector<vector<su2double> > Aux(2,vector<su2double>(2,0.0));
  vector<vector<su2double> > D(2,vector<su2double>(2,0.0));
  // Aux = M*Phi
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      Aux[i][j] = 0;
      for (int k=0; k<2; k++) {
        Aux[i][j] += M[i][k]*Phi[k][j];
      }
    }
  }

  // D = Phi'*Aux
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      D[i][j] = 0;
      for (int k=0; k<2; k++) {
        D[i][j] += Phi[k][i]*Aux[k][j]; //PHI transpose
      }
    }
  }

  //Modify the first column
  Phi[0][0] = Phi[0][0] * 1/sqrt(D[0][0]);
  Phi[1][0] = Phi[1][0] * 1/sqrt(D[0][0]);
  //Modify the second column
  Phi[0][1] = Phi[0][1] * 1/sqrt(D[1][1]);
  Phi[1][1] = Phi[1][1] * 1/sqrt(D[1][1]);

  // Sqrt of the eigenvalues (frequency of vibration of the modes)
  omega[0] = sqrt(Omega2[0][0]);
  omega[1] = sqrt(Omega2[1][1]);

}

void CSolver::SolveTypicalSectionWingModel(CGeometry *geometry, su2double Cl, su2double Cm, CConfig *config, unsigned short iMarker, vector<su2double>& displacements) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/

  /*--- Retrieve values from the config file ---*/
  su2double w_alpha = config->GetAeroelastic_Frequency_Pitch();
  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double dt      = config->GetDelta_UnstTimeND();
  dt = dt*w_alpha; //Non-dimensionalize the structural time.

  /*--- Structural Equation damping ---*/
  vector<su2double> xi(2,0.0);

  /*--- Eigenvectors and Eigenvalues of the Generalized EigenValue Problem. ---*/
  vector<vector<su2double> > Phi(2,vector<su2double>(2,0.0));   // generalized eigenvectors.
  vector<su2double> w(2,0.0);        // sqrt of the generalized eigenvalues (frequency of vibration of the modes).
  SetUpTypicalSectionWingModel(Phi, w, config);

  /*--- Solving the Decoupled Aeroelastic Problem with second order time discretization Eq (9) ---*/

  /*--- Solution variables description. //x[j][i], j-entry, i-equation. // Time (n+1)->np1, n->n, (n-1)->n1 ---*/
  vector<vector<su2double> > x_np1(2,vector<su2double>(2,0.0));

  /*--- Values from previous movement of spring at true time step n+1
   We use this values because we are solving for delta changes not absolute changes ---*/
  vector<vector<su2double> > x_np1_old = config->GetAeroelastic_np1(iMarker);

  /*--- Values at previous timesteps. ---*/
  vector<vector<su2double> > x_n = config->GetAeroelastic_n(iMarker);
  vector<vector<su2double> > x_n1 = config->GetAeroelastic_n1(iMarker);

  /*--- Set up of variables used to solve the structural problem. ---*/
  vector<su2double> f_tilde(2,0.0);
  vector<vector<su2double> > A_inv(2,vector<su2double>(2,0.0));
  su2double detA;
  su2double s1, s2;
  vector<su2double> rhs(2,0.0); //right hand side
  vector<su2double> eta(2,0.0);
  vector<su2double> eta_dot(2,0.0);

  /*--- Forcing Term ---*/
  su2double cons = vf*vf/PI_NUMBER;
  vector<su2double> f(2,0.0);
  f[0] = cons*(-Cl);
  f[1] = cons*(2*-Cm);

  //f_tilde = Phi'*f
  for (int i=0; i<2; i++) {
    f_tilde[i] = 0;
    for (int k=0; k<2; k++) {
      f_tilde[i] += Phi[k][i]*f[k]; //PHI transpose
    }
  }

  /*--- solve each decoupled equation (The inverse of the 2x2 matrix is provided) ---*/
  for (int i=0; i<2; i++) {
    /* Matrix Inverse */
    detA = 9.0/(4.0*dt*dt) + 3*w[i]*xi[i]/(dt) + w[i]*w[i];
    A_inv[0][0] = 1/detA * (3/(2.0*dt) + 2*xi[i]*w[i]);
    A_inv[0][1] = 1/detA * 1;
    A_inv[1][0] = 1/detA * -w[i]*w[i];
    A_inv[1][1] = 1/detA * 3/(2.0*dt);

    /* Source Terms from previous iterations */
    s1 = (-4*x_n[0][i] + x_n1[0][i])/(2.0*dt);
    s2 = (-4*x_n[1][i] + x_n1[1][i])/(2.0*dt);

    /* Problem Right Hand Side */
    rhs[0] = -s1;
    rhs[1] = f_tilde[i]-s2;

    /* Solve the equations */
    x_np1[0][i] = A_inv[0][0]*rhs[0] + A_inv[0][1]*rhs[1];
    x_np1[1][i] = A_inv[1][0]*rhs[0] + A_inv[1][1]*rhs[1];

    eta[i] = x_np1[0][i]-x_np1_old[0][i];  // For displacements, the change(deltas) is used.
    eta_dot[i] = x_np1[1][i]; // For velocities, absolute values are used.
  }

  // cout << "xn1      = [" << x_n1[0][0]      << ", " << x_n1[0][1]      << "]" << endl;
  // cout << "xn       = [" << x_n[0][0]       << ", " << x_n[0][1]       << "]" << endl;
  // cout << "xnp1_old = [" << x_np1_old[0][0] << ", " << x_np1_old[0][1] << "]" << endl;
  // cout << "xnp1     = [" << x_np1[0][0]     << ", " << x_np1[0][1]     << "]" << endl;

  /*--- Transform back from the generalized coordinates to get the actual displacements in plunge and pitch  q = Phi*eta ---*/
  vector<su2double> q(2,0.0);
  vector<su2double> q_dot(2,0.0);
  for (int i=0; i<2; i++) {
    q[i] = 0;
    q_dot[i] = 0;
    for (int k=0; k<2; k++) {
      q[i] += Phi[i][k]*eta[k];
      q_dot[i] += Phi[i][k]*eta_dot[k];
    }
  }

  su2double dh = b*q[0];
  su2double dalpha = q[1];

  su2double h_dot = w_alpha*b*q_dot[0];  //The w_a brings it back to actual time.
  su2double alpha_dot = w_alpha*q_dot[1];

  /*--- Set the solution of the structural equations ---*/
  displacements[0] = dh;
  displacements[1] = dalpha;
  displacements[2] = h_dot;
  displacements[3] = alpha_dot;

  /*--- Calculate the total plunge and total pitch displacements for the unsteady step by summing the displacement at each sudo time step ---*/
//su2double pitch, plunge;
//pitch = config->GetAeroelastic_pitch(iMarker);
//plunge = config->GetAeroelastic_plunge(iMarker);

//config->SetAeroelastic_pitch(iMarker , pitch+dalpha);
//config->SetAeroelastic_plunge(iMarker , plunge+dh/b);

  /*--- Set the Aeroelastic solution at time n+1. This gets update every sudo time step
   and after convering the sudo time step the solution at n+1 get moved to the solution at n
   in SetDualTime_Solver method ---*/

  config->SetAeroelastic_np1(iMarker, x_np1);

}

void CSolver::SolveModalWing(CGeometry *geometry, CConfig *config, unsigned short iMarker, su2double*& gen_forces, su2double*& gen_displacements) {

  /*--- The aeroelastic model solved in this routine is the typical section wing model
   The details of the implementation are similar to those found in J.J. Alonso
   "Fully-Implicit Time-Marching Aeroelastic Solutions" 1994. ---*/
 
  su2double Alpha = config->GetAoA()*PI_NUMBER/180.0;
  unsigned short modes = config->GetNumber_Modes();
  vector<su2double> w_modes(modes,0.0); //contains solution(displacements and rates) of typical section wing model.

  su2double vf      = config->GetAeroelastic_Flutter_Speed_Index();
  su2double b       = config->GetLength_Reynolds()/2.0; // airfoil semichord, Reynolds length is by defaul 1.0
  su2double Vo      = config->GetConicalRefVol();
  su2double dt      = config->GetDelta_UnstTimeND();

  su2double scale_param = 1/config->Get_Scaling_Parameter();

  //su2double scale_param = 1.0/175.125;

//   cout << "Inside Modal" << endl;


  su2double w_a = config->GetAeroelastic_Frequency_Pitch();
  /*--- Structural Equation damping ---*/
  vector<su2double> xi(modes,0.02);

  /*--- Solving the Decoupled Aeroelastic Problem with second order time discretization Eq (9) ---*/

  /*--- Solution variables description. //x[j][i], j-entry, i-equation. // Time (n+1)->np1, n->n, (n-1)->n1 ---*/
  vector<vector<su2double> > x_np1(2, vector<su2double>(modes,0.0));

  /*--- Values from previous movement of spring at true time step n+1
   We use this values because we are solving for delta changes not absolute changes ---*/
  vector<vector<su2double> > x_np1_old = config->GetAeroelastic_np1(iMarker);

  /*--- Values at previous timesteps. ---*/
  vector<vector<su2double> > x_n = config->GetAeroelastic_n(iMarker);
  vector<vector<su2double> > x_n1 = config->GetAeroelastic_n1(iMarker);

 // cout << "OLD SOL READ" << endl;
  /*--- Set up of variables used to solve the structural problem. ---*/
  vector<su2double> f_tilde(modes,0.0);
  vector<vector<su2double> > A_inv(2,vector<su2double>(2,0.0));
  su2double detA;
  su2double s1, s2;
  vector<su2double> rhs(2,0.0); //right hand side
  vector<su2double> eta(modes,0.0);
  vector<su2double> eta_dot(modes,0.0);

  /*--- Forcing Term ---*/
  su2double cons = vf*vf*b*b/2.0/Vo;

  for (unsigned short i=0;i<modes; i++) {

          w_modes[i] = config->GetAero_Omega(i)/w_a;

  }

  su2double value;
  cout << "F= [" ;
  for (unsigned short i=0;i<modes;i++) {  

       value = gen_forces[i];       
       f_tilde[i] = (cons * scale_param * value);
       cout << f_tilde[i] << " ";
  }
  cout << "]" << endl;

  dt = dt*w_a; //Non-dimensionalize the structural time.

//for (unsigned short i=0;i<modes;i++) {  
//     w_modes[i] = w_modes[i]/w_modes[1];
//}
  /*--- solve each decoupled equation (The inverse of the 2x2 matrix is provided) ---*/
  for (unsigned short i=0; i<modes; i++) {
    /* Matrix Inverse */
//	  cout << "Mode : " << i << endl; 

    detA = 9.0/(4.0*dt*dt) + 3*w_modes[i]*xi[i]/(dt) + w_modes[i]*w_modes[i];
    A_inv[0][0] = 1/detA * (3/(2.0*dt) + 2*xi[i]*w_modes[i]);
    A_inv[0][1] = 1/detA * 1;
    A_inv[1][0] = 1/detA * -w_modes[i]*w_modes[i];
    A_inv[1][1] = 1/detA * 3/(2.0*dt);

    /* Source Terms from previous iterations */
    s1 = (-4*x_n[0][i] + x_n1[0][i])/(2.0*dt);
    s2 = (-4*x_n[1][i] + x_n1[1][i])/(2.0*dt);

    /* Problem Right Hand Side */
    rhs[0] = -s1;
    rhs[1] = f_tilde[i]-s2;

    /* Solve the equations */
    x_np1[0][i] = A_inv[0][0]*rhs[0] + A_inv[0][1]*rhs[1];
    x_np1[1][i] = A_inv[1][0]*rhs[0] + A_inv[1][1]*rhs[1];

    eta[i] = x_np1[0][i]-x_np1_old[0][i];  // For displacements, the change(deltas) is used.
    eta_dot[i] = x_np1[1][i]; // For velocities, absolute values are used.
  }
  /*--- Transform back from the generalized coordinates to get the actual displacements in plunge and pitch  q = Phi*eta ---*/
  /*--- Set the solution of the structural equations ---*/
 
  cout << " xnp_old= [" ;
  for (unsigned short i=0;i<modes;i++) {  

       cout << x_np1_old[0][i] << " ";
  }
  cout << "]" << endl;
  cout << " xnp= [" ;
  for (unsigned short i=0;i<modes;i++) {  

       cout << x_np1[0][i] << " ";
  }
  cout << "]" << endl;


   // cout << "Setting solution" << endl;
  gen_displacements[0] = eta[0];
  gen_displacements[1] = eta[1];
  gen_displacements[2] = eta[2];
  gen_displacements[3] = eta[3];

  /*--- Calculate the total plunge and total pitch displacements for the unsteady step by summing the displacement at each sudo time step ---*/
  su2double pitch, plunge;
  pitch = config->GetAeroelastic_pitch(iMarker);
  plunge = config->GetAeroelastic_plunge(iMarker);

 // cout << "about to store" << endl;
  
  config->SetAeroelastic_pitch(iMarker , pitch+gen_displacements[0]);
  config->SetAeroelastic_plunge(iMarker , plunge+gen_displacements[1]);


  //cout << "Mode1: " << pitch+gen_displacements[0] << " | Mode2: " << plunge+gen_displacements[1] << endl;
 // cout << "pitch plunge stored" << endl;
  /*--- Set the Aeroelastic solution at time n+1. This gets update every sudo time step
   and after convering the sudo time step the solution at n+1 get moved to the solution at n
   in SetDualTime_Solver method ---*/

  config->SetAeroelastic_np1(iMarker, x_np1);

 // cout << "NEW SOL SET" << endl;
}

void CSolver::Restart_OldGeometry(CGeometry *geometry, CConfig *config) {

  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {

  /*--- This function is intended for dual time simulations ---*/

  int Unst_RestartIter;
  ifstream restart_file_n;

  string filename = config->GetSolution_FileName();
  string filename_n;

  /*--- Auxiliary vector for storing the coordinates ---*/
  su2double Coord[3] = {0.0};

  /*--- Variables for reading the restart files ---*/
  string text_line;
  long iPoint_Local;
  unsigned long iPoint_Global_Local = 0, iPoint_Global = 0;

  /*--- First, we load the restart file for time n ---*/

  /*-------------------------------------------------------------------------------------------*/

  /*--- Modify file name for an unsteady restart ---*/
  if (config->GetRestart()) Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter())-1;
  else Unst_RestartIter = SU2_TYPE::Int(config->GetUnst_AdjointIter())-1;
  filename_n = config->GetFilename(filename, ".csv", Unst_RestartIter);

  /*--- Open the restart file, throw an error if this fails. ---*/

  restart_file_n.open(filename_n.data(), ios::in);
  if (restart_file_n.fail()) {
    SU2_MPI::Error(string("There is no flow restart file ") + filename_n, CURRENT_FUNCTION);
  }

  /*--- First, set all indices to a negative value by default, and Global n indices to 0 ---*/
  iPoint_Global_Local = 0; iPoint_Global = 0;

  /*--- Read all lines in the restart file ---*/
  /*--- The first line is the header ---*/

  getline (restart_file_n, text_line);

  for (iPoint_Global = 0; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++ ) {

    getline (restart_file_n, text_line);

    vector<string> point_line = PrintingToolbox::split(text_line, ',');

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local > -1) {

      Coord[0] = PrintingToolbox::stod(point_line[1]);
      Coord[1] = PrintingToolbox::stod(point_line[2]);
      if (nDim == 3){
        Coord[2] = PrintingToolbox::stod(point_line[3]);
      }
      geometry->nodes->SetCoord_n(iPoint_Local, Coord);

      iPoint_Global_Local++;
    }
  }

  /*--- Detect a wrong solution file ---*/

  if (iPoint_Global_Local < geometry->GetnPointDomain()) {
    SU2_MPI::Error(string("The solution file ") + filename + string(" doesn't match with the mesh file!\n") +
                   string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
  }

  /*--- Close the restart file ---*/

  restart_file_n.close();

  /*-------------------------------------------------------------------------------------------*/
  /*-------------------------------------------------------------------------------------------*/

  /*--- Now, we load the restart file for time n-1, if the simulation is 2nd Order ---*/

  if (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND) {

    ifstream restart_file_n1;
    string filename_n1;

    /*--- Modify file name for an unsteady restart ---*/
    if (config->GetRestart()) Unst_RestartIter = SU2_TYPE::Int(config->GetRestart_Iter())-2;
    else Unst_RestartIter = SU2_TYPE::Int(config->GetUnst_AdjointIter())-2;
    filename_n1 = config->GetFilename(filename, ".csv", Unst_RestartIter);

    /*--- Open the restart file, throw an error if this fails. ---*/

    restart_file_n1.open(filename_n1.data(), ios::in);
    if (restart_file_n1.fail()) {
        SU2_MPI::Error(string("There is no flow restart file ") + filename_n1, CURRENT_FUNCTION);

    }

    /*--- First, set all indices to a negative value by default, and Global n indices to 0 ---*/
    iPoint_Global_Local = 0; iPoint_Global = 0;

    /*--- Read all lines in the restart file ---*/
    /*--- The first line is the header ---*/

    getline (restart_file_n1, text_line);

    for (iPoint_Global = 0; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++ ) {

      getline (restart_file_n1, text_line);

      vector<string> point_line = PrintingToolbox::split(text_line, ',');

      /*--- Retrieve local index. If this node from the restart file lives
       on the current processor, we will load and instantiate the vars. ---*/

      iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

      if (iPoint_Local > -1) {

        Coord[0] = PrintingToolbox::stod(point_line[1]);
        Coord[1] = PrintingToolbox::stod(point_line[2]);
        if (nDim == 3){
          Coord[2] = PrintingToolbox::stod(point_line[3]);
        }

        geometry->nodes->SetCoord_n1(iPoint_Local, Coord);

        iPoint_Global_Local++;
      }

    }

    /*--- Detect a wrong solution file ---*/

    if (iPoint_Global_Local < geometry->GetnPointDomain()) {
      SU2_MPI::Error(string("The solution file ") + filename + string(" doesn't match with the mesh file!\n") +
                     string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
    }

    /*--- Close the restart file ---*/

    restart_file_n1.close();

  }

  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS

  /*--- It's necessary to communicate this information ---*/

  geometry->InitiateComms(geometry, config, COORDINATES_OLD);
  geometry->CompleteComms(geometry, config, COORDINATES_OLD);

}

void CSolver::Read_SU2_Restart_ASCII(CGeometry *geometry, const CConfig *config, string val_filename) {

  ifstream restart_file;
  string text_line, Tag;
  unsigned short iVar;
  long iPoint_Local = 0; unsigned long iPoint_Global = 0;
  int counter = 0;
  fields.clear();

  Restart_Vars = new int[5];

  string error_string = "Note: ASCII restart files must be in CSV format since v7.0.\n"
                        "Check https://su2code.github.io/docs/Guide-to-v7 for more information.";

  /*--- First, check that this is not a binary restart file. ---*/

  char fname[100];
  val_filename += ".csv";
  strcpy(fname, val_filename.c_str());
  int magic_number;

#ifndef HAVE_MPI

  /*--- Serial binary input. ---*/

  FILE *fhw;
  fhw = fopen(fname,"rb");
  size_t ret;

  /*--- Error check for opening the file. ---*/

  if (!fhw) {
    SU2_MPI::Error(string("Unable to open SU2 restart file ") + fname, CURRENT_FUNCTION);
  }

  /*--- Attempt to read the first int, which should be our magic number. ---*/

  ret = fread(&magic_number, sizeof(int), 1, fhw);
  if (ret != 1) {
    SU2_MPI::Error("Error reading restart file.", CURRENT_FUNCTION);
  }

  /*--- Check that this is an SU2 binary file. SU2 binary files
   have the hex representation of "SU2" as the first int in the file. ---*/

  if (magic_number == 535532) {
    SU2_MPI::Error(string("File ") + string(fname) + string(" is a binary SU2 restart file, expected ASCII.\n") +
                   string("SU2 reads/writes binary restart files by default.\n") +
                   string("Note that backward compatibility for ASCII restart files is\n") +
                   string("possible with the READ_BINARY_RESTART option."), CURRENT_FUNCTION);
  }

  fclose(fhw);

#else

  /*--- Parallel binary input using MPI I/O. ---*/

  MPI_File fhw;
  int ierr;

  /*--- All ranks open the file using MPI. ---*/

  ierr = MPI_File_open(SU2_MPI::GetComm(), fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhw);

  /*--- Error check opening the file. ---*/

  if (ierr) {
    SU2_MPI::Error(string("SU2 ASCII restart file ") + string(fname) + string(" not found.\n") + error_string,
                   CURRENT_FUNCTION);
  }

  /*--- Have the master attempt to read the magic number. ---*/

  if (rank == MASTER_NODE)
    MPI_File_read(fhw, &magic_number, 1, MPI_INT, MPI_STATUS_IGNORE);

  /*--- Broadcast the number of variables to all procs and store clearly. ---*/

  SU2_MPI::Bcast(&magic_number, 1, MPI_INT, MASTER_NODE, SU2_MPI::GetComm());

  /*--- Check that this is an SU2 binary file. SU2 binary files
   have the hex representation of "SU2" as the first int in the file. ---*/

  if (magic_number == 535532) {
    SU2_MPI::Error(string("File ") + string(fname) + string(" is a binary SU2 restart file, expected ASCII.\n") +
                   string("SU2 reads/writes binary restart files by default.\n") +
                   string("Note that backward compatibility for ASCII restart files is\n") +
                   string("possible with the READ_BINARY_RESTART option."), CURRENT_FUNCTION);
  }

  MPI_File_close(&fhw);

#endif

  /*--- Open the restart file ---*/

  restart_file.open(val_filename.data(), ios::in);

  /*--- In case there is no restart file ---*/

  if (restart_file.fail()) {
    SU2_MPI::Error(string("SU2 ASCII restart file ") + string(fname) + string(" not found.\n") + error_string,
                   CURRENT_FUNCTION);
  }

  /*--- Identify the number of fields (and names) in the restart file ---*/

  getline (restart_file, text_line);

  char delimiter = ',';
  fields = PrintingToolbox::split(text_line, delimiter);

  if (fields.size() <= 1) {
    SU2_MPI::Error(string("Restart file does not seem to be a CSV file.\n") + error_string, CURRENT_FUNCTION);
  }

  for (unsigned short iField = 0; iField < fields.size(); iField++){
    PrintingToolbox::trim(fields[iField]);
  }

  /*--- Set the number of variables, one per field in the
   restart file (without including the PointID) ---*/

  Restart_Vars[1] = (int)fields.size() - 1;

  /*--- Allocate memory for the restart data. ---*/

  Restart_Data = new passivedouble[Restart_Vars[1]*geometry->GetnPointDomain()];

  /*--- Read all lines in the restart file and extract data. ---*/

  for (iPoint_Global = 0; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++) {

    if (!getline (restart_file, text_line)) break;

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local > -1) {

      vector<string> point_line = PrintingToolbox::split(text_line, delimiter);

      /*--- Store the solution (starting with node coordinates) --*/

      for (iVar = 0; iVar < Restart_Vars[1]; iVar++)
        Restart_Data[counter*Restart_Vars[1] + iVar] = SU2_TYPE::GetValue(PrintingToolbox::stod(point_line[iVar+1]));

      /*--- Increment our local point counter. ---*/

      counter++;

    }
  }

  if (iPoint_Global != geometry->GetGlobal_nPointDomain())
    SU2_MPI::Error("The solution file does not match the mesh, currently only binary files can be interpolated.",
                   CURRENT_FUNCTION);

}

void CSolver::Read_SU2_Restart_Binary(CGeometry *geometry, const CConfig *config, string val_filename) {

  char str_buf[CGNS_STRING_SIZE], fname[100];
  val_filename += ".dat";
  strcpy(fname, val_filename.c_str());
  const int nRestart_Vars = 5;
  Restart_Vars = new int[nRestart_Vars];
  fields.clear();

#ifndef HAVE_MPI

  /*--- Serial binary input. ---*/

  FILE *fhw;
  fhw = fopen(fname,"rb");
  size_t ret;

  /*--- Error check for opening the file. ---*/

  if (!fhw) {
    SU2_MPI::Error(string("Unable to open SU2 restart file ") + string(fname), CURRENT_FUNCTION);
  }

  /*--- First, read the number of variables and points. ---*/

  ret = fread(Restart_Vars, sizeof(int), nRestart_Vars, fhw);
  if (ret != (unsigned long)nRestart_Vars) {
    SU2_MPI::Error("Error reading restart file.", CURRENT_FUNCTION);
  }

  /*--- Check that this is an SU2 binary file. SU2 binary files
   have the hex representation of "SU2" as the first int in the file. ---*/

  if (Restart_Vars[0] != 535532) {
    SU2_MPI::Error(string("File ") + string(fname) + string(" is not a binary SU2 restart file.\n") +
                   string("SU2 reads/writes binary restart files by default.\n") +
                   string("Note that backward compatibility for ASCII restart files is\n") +
                   string("possible with the READ_BINARY_RESTART option."), CURRENT_FUNCTION);
  }

  /*--- Store the number of fields and points to be read for clarity. ---*/

  const unsigned long nFields = Restart_Vars[1];
  const unsigned long nPointFile = Restart_Vars[2];

  /*--- Read the variable names from the file. Note that we are adopting a
   fixed length of 33 for the string length to match with CGNS. This is
   needed for when we read the strings later. We pad the beginning of the
   variable string vector with the Point_ID tag that wasn't written. ---*/

  fields.push_back("Point_ID");
  for (auto iVar = 0u; iVar < nFields; iVar++) {
    ret = fread(str_buf, sizeof(char), CGNS_STRING_SIZE, fhw);
    if (ret != (unsigned long)CGNS_STRING_SIZE) {
      SU2_MPI::Error("Error reading restart file.", CURRENT_FUNCTION);
    }
    fields.push_back(str_buf);
  }

  /*--- For now, create a temp 1D buffer to read the data from file. ---*/

  Restart_Data = new passivedouble[nFields*nPointFile];

  /*--- Read in the data for the restart at all local points. ---*/

  ret = fread(Restart_Data, sizeof(passivedouble), nFields*nPointFile, fhw);
  if (ret != nFields*nPointFile) {
    SU2_MPI::Error("Error reading restart file.", CURRENT_FUNCTION);
  }

  /*--- Close the file. ---*/

  fclose(fhw);

#else

  /*--- Parallel binary input using MPI I/O. ---*/

  MPI_File fhw;
  SU2_MPI::Status status;
  MPI_Datatype etype, filetype;
  MPI_Offset disp;

  /*--- All ranks open the file using MPI. ---*/

  int ierr = MPI_File_open(SU2_MPI::GetComm(), fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhw);

  if (ierr) SU2_MPI::Error(string("Unable to open SU2 restart file ") + string(fname), CURRENT_FUNCTION);

  /*--- First, read the number of variables and points (i.e., cols and rows),
   which we will need in order to read the file later. Also, read the
   variable string names here. Only the master rank reads the header. ---*/

  if (rank == MASTER_NODE)
    MPI_File_read(fhw, Restart_Vars, nRestart_Vars, MPI_INT, MPI_STATUS_IGNORE);

  /*--- Broadcast the number of variables to all procs and store clearly. ---*/

  SU2_MPI::Bcast(Restart_Vars, nRestart_Vars, MPI_INT, MASTER_NODE, SU2_MPI::GetComm());

  /*--- Check that this is an SU2 binary file. SU2 binary files
   have the hex representation of "SU2" as the first int in the file. ---*/

  if (Restart_Vars[0] != 535532) {
    SU2_MPI::Error(string("File ") + string(fname) + string(" is not a binary SU2 restart file.\n") +
                   string("SU2 reads/writes binary restart files by default.\n") +
                   string("Note that backward compatibility for ASCII restart files is\n") +
                   string("possible with the READ_BINARY_RESTART option."), CURRENT_FUNCTION);
  }

  /*--- Store the number of fields and points to be read for clarity. ---*/

  const unsigned long nFields = Restart_Vars[1];
  const unsigned long nPointFile = Restart_Vars[2];

  /*--- Read the variable names from the file. Note that we are adopting a
   fixed length of 33 for the string length to match with CGNS. This is
   needed for when we read the strings later. ---*/

  char *mpi_str_buf = new char[nFields*CGNS_STRING_SIZE];
  if (rank == MASTER_NODE) {
    disp = nRestart_Vars*sizeof(int);
    MPI_File_read_at(fhw, disp, mpi_str_buf, nFields*CGNS_STRING_SIZE,
                     MPI_CHAR, MPI_STATUS_IGNORE);
  }

  /*--- Broadcast the string names of the variables. ---*/

  SU2_MPI::Bcast(mpi_str_buf, nFields*CGNS_STRING_SIZE, MPI_CHAR,
                 MASTER_NODE, SU2_MPI::GetComm());

  /*--- Now parse the string names and load into the config class in case
   we need them for writing visualization files (SU2_SOL). ---*/

  fields.push_back("Point_ID");
  for (auto iVar = 0u; iVar < nFields; iVar++) {
    const auto index = iVar*CGNS_STRING_SIZE;
    string field_buf("\"");
    for (int iChar = 0; iChar < CGNS_STRING_SIZE; iChar++) {
      str_buf[iChar] = mpi_str_buf[index + iChar];
    }
    field_buf.append(str_buf);
    field_buf.append("\"");
    fields.push_back(field_buf.c_str());
  }

  /*--- Free string buffer memory. ---*/

  delete [] mpi_str_buf;

  /*--- We're writing only su2doubles in the data portion of the file. ---*/

  etype = MPI_DOUBLE;

  /*--- We need to ignore the 4 ints describing the nVar_Restart and nPoints,
   along with the string names of the variables. ---*/

  disp = nRestart_Vars*sizeof(int) + CGNS_STRING_SIZE*nFields*sizeof(char);

  /*--- Define a derived datatype for this rank's set of non-contiguous data
   that will be placed in the restart. Here, we are collecting each one of the
   points which are distributed throughout the file in blocks of nVar_Restart data. ---*/

  int nBlock;
  int *blocklen = nullptr;
  MPI_Aint *displace = nullptr;

  if (nPointFile == geometry->GetGlobal_nPointDomain() ||
      config->GetKind_SU2() == SU2_COMPONENT::SU2_SOL) {
    /*--- No interpolation, each rank reads the indices it needs. ---*/
    nBlock = geometry->GetnPointDomain();

    blocklen = new int[nBlock];
    displace = new MPI_Aint[nBlock];
    int counter = 0;
    for (auto iPoint_Global = 0ul; iPoint_Global < geometry->GetGlobal_nPointDomain(); ++iPoint_Global) {
      if (geometry->GetGlobal_to_Local_Point(iPoint_Global) > -1) {
        blocklen[counter] = nFields;
        displace[counter] = iPoint_Global*nFields*sizeof(passivedouble);
        counter++;
      }
    }
  }
  else {
    /*--- Interpolation required, read large blocks of data. ---*/
    nBlock = 1;

    blocklen = new int[nBlock];
    displace = new MPI_Aint[nBlock];

    const auto partitioner = CLinearPartitioner(nPointFile,0);

    blocklen[0] = nFields*partitioner.GetSizeOnRank(rank);
    displace[0] = nFields*partitioner.GetFirstIndexOnRank(rank)*sizeof(passivedouble);;
  }

  MPI_Type_create_hindexed(nBlock, blocklen, displace, MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);

  /*--- Set the view for the MPI file write, i.e., describe the location in
   the file that this rank "sees" for writing its piece of the restart file. ---*/

  MPI_File_set_view(fhw, disp, etype, filetype, (char*)"native", MPI_INFO_NULL);

  /*--- For now, create a temp 1D buffer to read the data from file. ---*/

  const int bufSize = nBlock*blocklen[0];
  Restart_Data = new passivedouble[bufSize];

  /*--- Collective call for all ranks to read from their view simultaneously. ---*/

  MPI_File_read_all(fhw, Restart_Data, bufSize, MPI_DOUBLE, &status);

  /*--- All ranks close the file after writing. ---*/

  MPI_File_close(&fhw);

  /*--- Free the derived datatype and release temp memory. ---*/

  MPI_Type_free(&filetype);

  delete [] blocklen;
  delete [] displace;

#endif

  if (nPointFile != geometry->GetGlobal_nPointDomain() &&
      config->GetKind_SU2() != SU2_COMPONENT::SU2_SOL) {
    InterpolateRestartData(geometry, config);
  }
}

void CSolver::InterpolateRestartData(const CGeometry *geometry, const CConfig *config) {

  if (geometry->GetGlobal_nPointDomain() == 0) return;

  if (size != SINGLE_NODE && size % 2)
    SU2_MPI::Error("Number of ranks must be multiple of 2.", CURRENT_FUNCTION);

  if (config->GetFEMSolver())
    SU2_MPI::Error("Cannot interpolate the restart file for FEM problems.", CURRENT_FUNCTION);

  /* Challenges:
   *  - Do not use too much memory by gathering the restart data in all ranks.
   *  - Do not repeat too many computations in all ranks.
   * Solution?:
   *  - Build a local ADT for the domain points (not the restart points).
   *  - Find the closest target point for each donor, which does not match all targets.
   *  - "Diffuse" the data to neighbor points.
   *  Complexity is approx. Nlt + (Nlt + Nd) log(Nlt) where Nlt is the LOCAL number
   *  of target points and Nd the TOTAL number of donors. */

  const unsigned long nFields = Restart_Vars[1];
  const unsigned long nPointFile = Restart_Vars[2];
  const auto t0 = SU2_MPI::Wtime();
  auto nRecurse = 0;

  if (rank == MASTER_NODE) {
    cout << "\nThe number of points in the restart file (" << nPointFile << ") does not match "
            "the mesh (" << geometry->GetGlobal_nPointDomain() << ").\n"
            "A recursive nearest neighbor interpolation will be performed." << endl;
  }

  su2activematrix localVars(nPointDomain, nFields);
  localVars = su2double(0.0);
  {
  su2vector<uint8_t> isMapped(nPoint);
  isMapped = false;

  /*--- ADT of local target points. ---*/
  {
  const auto& coord = geometry->nodes->GetCoord();
  vector<unsigned long> index(nPointDomain);
  iota(index.begin(), index.end(), 0ul);

  CADTPointsOnlyClass adt(nDim, nPointDomain, coord.data(), index.data(), false);
  vector<unsigned long>().swap(index);

  /*--- Copy local donor restart data, which will circulate over all ranks. ---*/

  const auto partitioner = CLinearPartitioner(nPointFile,0);

  unsigned long nPointDonorMax = 0;
  for (int i=0; i<size; ++i)
    nPointDonorMax = max(nPointDonorMax, partitioner.GetSizeOnRank(i));

  su2activematrix sendBuf(nPointDonorMax, nFields);

  for (auto iPoint = 0ul; iPoint < nPointDonorMax; ++iPoint) {
    const auto iPointDonor = min(iPoint,partitioner.GetSizeOnRank(rank)-1ul);
    for (auto iVar = 0ul; iVar < nFields; ++iVar)
      sendBuf(iPoint,iVar) = Restart_Data[iPointDonor*nFields+iVar];
  }

  delete [] Restart_Data;
  Restart_Data = nullptr;

  /*--- Make room to receive donor data from other ranks, and to map it to target points. ---*/

  su2activematrix donorVars(nPointDonorMax, nFields);
  vector<su2double> donorDist(nPointDomain, 1e12);

  /*--- Circle over all ranks. ---*/

  const int dst = (rank+1) % size; // send to next
  const int src = (rank-1+size) % size; // receive from prev.
  const int count = sendBuf.size();

  for (int iStep = 0; iStep < size; ++iStep) {

    swap(sendBuf, donorVars);

    if (iStep) {
      /*--- Odd ranks send and then receive, and vice versa. ---*/
      if (rank%2) SU2_MPI::Send(sendBuf.data(), count, MPI_DOUBLE, dst, 0, SU2_MPI::GetComm());
      else SU2_MPI::Recv(donorVars.data(), count, MPI_DOUBLE, src, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);

      if (rank%2==0) SU2_MPI::Send(sendBuf.data(), count, MPI_DOUBLE, dst, 0, SU2_MPI::GetComm());
      else SU2_MPI::Recv(donorVars.data(), count, MPI_DOUBLE, src, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
    }

    /*--- Find the closest target for each donor. ---*/

    vector<su2double> targetDist(donorVars.rows());
    vector<unsigned long> iTarget(donorVars.rows());

    SU2_OMP_PARALLEL_(for schedule(dynamic,4*OMP_MIN_SIZE))
    for (auto iDonor = 0ul; iDonor < donorVars.rows(); ++iDonor) {
      int r=0;
      adt.DetermineNearestNode(donorVars[iDonor], targetDist[iDonor], iTarget[iDonor], r);
    }
    END_SU2_OMP_PARALLEL

    /*--- Keep the closest donor for each target (this is separate for OpenMP). ---*/

    for (auto iDonor = 0ul; iDonor < donorVars.rows(); ++iDonor) {
      const auto iPoint = iTarget[iDonor];
      const auto dist = targetDist[iDonor];

      if (dist < donorDist[iPoint]) {
        donorDist[iPoint] = dist;
        isMapped[iPoint] = true;
        for (auto iVar = 0ul; iVar < donorVars.cols(); ++iVar)
          localVars(iPoint,iVar) = donorVars(iDonor,iVar);
      }
    }
  }
  } // everything goes out of scope except "localVars" and "isMapped"

  /*--- Recursively diffuse the nearest neighbor data. ---*/

  auto nDonor = isMapped;
  bool done = false;

  SU2_OMP_PARALLEL
  while (!done) {
    SU2_OMP_FOR_DYN(roundUpDiv(nPointDomain,2*omp_get_num_threads()))
    for (auto iPoint = 0ul; iPoint < nPointDomain; ++iPoint) {
      /*--- Do not change points that are already interpolated. ---*/
      if (isMapped[iPoint]) continue;

      /*--- Boundaries to boundaries and domain to domain. ---*/
      const bool boundary_i = geometry->nodes->GetSolidBoundary(iPoint);

      for (const auto jPoint : geometry->nodes->GetPoints(iPoint)) {
        if (!isMapped[jPoint]) continue;
        if (boundary_i != geometry->nodes->GetSolidBoundary(jPoint)) continue;

        nDonor[iPoint]++;

        for (auto iVar = 0ul; iVar < localVars.cols(); ++iVar)
          localVars(iPoint,iVar) += localVars(jPoint,iVar);
      }

      if (nDonor[iPoint] > 0) {
        for (auto iVar = 0ul; iVar < localVars.cols(); ++iVar)
          localVars(iPoint,iVar) /= nDonor[iPoint];
        nDonor[iPoint] = true;
      }
    }
    END_SU2_OMP_FOR

    /*--- Repeat while all points are not mapped. ---*/

    SU2_OMP_MASTER {
      done = true;
      ++nRecurse;
    }
    END_SU2_OMP_MASTER

    bool myDone = true;

    SU2_OMP_FOR_STAT(16*OMP_MIN_SIZE)
    for (auto iPoint = 0ul; iPoint < nPointDomain; ++iPoint) {
      isMapped[iPoint] = nDonor[iPoint];
      myDone &= nDonor[iPoint];
    }
    END_SU2_OMP_FOR

    SU2_OMP_ATOMIC
    done &= myDone;

    SU2_OMP_BARRIER
  }
  END_SU2_OMP_PARALLEL

  } // everything goes out of scope except "localVars"

  /*--- Move to Restart_Data in ascending order of global index, which is how a matching restart would have been read. ---*/

  Restart_Data = new passivedouble[nPointDomain*nFields];
  Restart_Vars[2] = nPointDomain;

  int counter = 0;
  for (auto iPoint_Global = 0ul; iPoint_Global < geometry->GetGlobal_nPointDomain(); ++iPoint_Global) {
    const auto iPoint = geometry->GetGlobal_to_Local_Point(iPoint_Global);
    if (iPoint >= 0) {
      for (auto iVar = 0ul; iVar < nFields; ++iVar)
        Restart_Data[counter*nFields+iVar] = SU2_TYPE::GetValue(localVars(iPoint,iVar));
      counter++;
    }
  }

  if (rank == MASTER_NODE) {
    cout << "Number of recursions: " << nRecurse << ".\n"
            "Elapsed time: " << SU2_MPI::Wtime()-t0 << "s.\n" << endl;
  }
}

void CSolver::Read_SU2_Restart_Metadata(CGeometry *geometry, CConfig *config, bool adjoint, string val_filename) const {

  su2double AoA_ = config->GetAoA();
  su2double AoS_ = config->GetAoS();
  su2double BCThrust_ = config->GetInitial_BCThrust();
  su2double dCD_dCL_ = config->GetdCD_dCL();
  su2double dCMx_dCL_ = config->GetdCMx_dCL();
  su2double dCMy_dCL_ = config->GetdCMy_dCL();
  su2double dCMz_dCL_ = config->GetdCMz_dCL();
  su2double SPPressureDrop_ = config->GetStreamwise_Periodic_PressureDrop();
  string::size_type position;
  unsigned long InnerIter_ = 0;
  ifstream restart_file;

  /*--- Carry on with ASCII metadata reading. ---*/

  restart_file.open(val_filename.data(), ios::in);
  if (restart_file.fail()) {
    if (rank == MASTER_NODE) {
      cout << " Warning: There is no restart file (" << val_filename.data() << ")."<< endl;
      cout << " Computation will continue without updating metadata parameters." << endl;
    }
  }
  else {

    string text_line;

    /*--- Space for extra info (if any) ---*/

    while (getline (restart_file, text_line)) {

      /*--- External iteration ---*/

      position = text_line.find ("ITER=",0);
      if (position != string::npos) {
        // TODO: 'ITER=' has 5 chars, not 9!
        text_line.erase (0,9); InnerIter_ = atoi(text_line.c_str());
      }

      /*--- Angle of attack ---*/

      position = text_line.find ("AOA=",0);
      if (position != string::npos) {
        text_line.erase (0,4); AoA_ = atof(text_line.c_str());
      }

      /*--- Sideslip angle ---*/

      position = text_line.find ("SIDESLIP_ANGLE=",0);
      if (position != string::npos) {
        text_line.erase (0,15); AoS_ = atof(text_line.c_str());
      }

      /*--- BCThrust angle ---*/

      position = text_line.find ("INITIAL_BCTHRUST=",0);
      if (position != string::npos) {
        text_line.erase (0,17); BCThrust_ = atof(text_line.c_str());
      }

      /*--- dCD_dCL coefficient ---*/

      position = text_line.find ("DCD_DCL_VALUE=",0);
      if (position != string::npos) {
        text_line.erase (0,14); dCD_dCL_ = atof(text_line.c_str());
      }

      /*--- dCMx_dCL coefficient ---*/

      position = text_line.find ("DCMX_DCL_VALUE=",0);
      if (position != string::npos) {
        text_line.erase (0,15); dCMx_dCL_ = atof(text_line.c_str());
      }

      /*--- dCMy_dCL coefficient ---*/

      position = text_line.find ("DCMY_DCL_VALUE=",0);
      if (position != string::npos) {
        text_line.erase (0,15); dCMy_dCL_ = atof(text_line.c_str());
      }

      /*--- dCMz_dCL coefficient ---*/

      position = text_line.find ("DCMZ_DCL_VALUE=",0);
      if (position != string::npos) {
        text_line.erase (0,15); dCMz_dCL_ = atof(text_line.c_str());
      }

      /*--- Streamwise periodic pressure drop for prescribed massflow cases. ---*/

      position = text_line.find ("STREAMWISE_PERIODIC_PRESSURE_DROP=",0);
      if (position != string::npos) {
        // Erase the name from the line, 'STREAMWISE_PERIODIC_PRESSURE_DROP=' has 34 chars.
        text_line.erase (0,34); SPPressureDrop_ = atof(text_line.c_str());
      }

    }

    /*--- Close the restart meta file. ---*/

    restart_file.close();

  }


  /*--- Load the metadata. ---*/

  /*--- Angle of attack ---*/

  if (config->GetDiscard_InFiles() == false) {
    if ((config->GetAoA() != AoA_) && (rank == MASTER_NODE)) {
      cout.precision(6);
      cout <<"WARNING: AoA in the solution file (" << AoA_ << " deg.) +" << endl;
      cout << "         AoA offset in mesh file (" << config->GetAoA_Offset() << " deg.) = " << AoA_ + config->GetAoA_Offset() << " deg." << endl;
    }
    config->SetAoA(AoA_ + config->GetAoA_Offset());
  }

  else {
    if ((config->GetAoA() != AoA_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the AoA in the solution file." << endl;
  }

  /*--- Sideslip angle ---*/

  if (config->GetDiscard_InFiles() == false) {
    if ((config->GetAoS() != AoS_) && (rank == MASTER_NODE)) {
      cout.precision(6);
      cout <<"WARNING: AoS in the solution file (" << AoS_ << " deg.) +" << endl;
      cout << "         AoS offset in mesh file (" << config->GetAoS_Offset() << " deg.) = " << AoS_ + config->GetAoS_Offset() << " deg." << endl;
    }
    config->SetAoS(AoS_ + config->GetAoS_Offset());
  }
  else {
    if ((config->GetAoS() != AoS_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the AoS in the solution file." << endl;
  }

  /*--- BCThrust ---*/

  if (config->GetDiscard_InFiles() == false) {
    if ((config->GetInitial_BCThrust() != BCThrust_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the initial BC Thrust provided in the solution file: " << BCThrust_ << " lbs." << endl;
    config->SetInitial_BCThrust(BCThrust_);
  }
  else {
    if ((config->GetInitial_BCThrust() != BCThrust_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the BC Thrust in the solution file." << endl;
  }


  if (config->GetDiscard_InFiles() == false) {

    if ((config->GetdCD_dCL() != dCD_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the dCD/dCL provided in the direct solution file: " << dCD_dCL_ << "." << endl;
    config->SetdCD_dCL(dCD_dCL_);

    if ((config->GetdCMx_dCL() != dCMx_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the dCMx/dCL provided in the direct solution file: " << dCMx_dCL_ << "." << endl;
    config->SetdCMx_dCL(dCMx_dCL_);

    if ((config->GetdCMy_dCL() != dCMy_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the dCMy/dCL provided in the direct solution file: " << dCMy_dCL_ << "." << endl;
    config->SetdCMy_dCL(dCMy_dCL_);

    if ((config->GetdCMz_dCL() != dCMz_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the dCMz/dCL provided in the direct solution file: " << dCMz_dCL_ << "." << endl;
    config->SetdCMz_dCL(dCMz_dCL_);

  }

  else {

    if ((config->GetdCD_dCL() != dCD_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the dCD/dCL in the direct solution file." << endl;

    if ((config->GetdCMx_dCL() != dCMx_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the dCMx/dCL in the direct solution file." << endl;

    if ((config->GetdCMy_dCL() != dCMy_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the dCMy/dCL in the direct solution file." << endl;

    if ((config->GetdCMz_dCL() != dCMz_dCL_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the dCMz/dCL in the direct solution file." << endl;

  }

  if (config->GetDiscard_InFiles() == false) {
    if ((config->GetStreamwise_Periodic_PressureDrop() != SPPressureDrop_) && (rank == MASTER_NODE))
      cout <<"WARNING: SU2 will use the STREAMWISE_PERIODIC_PRESSURE_DROP provided in the direct solution file: " << std::setprecision(16) << SPPressureDrop_ << endl;
    config->SetStreamwise_Periodic_PressureDrop(SPPressureDrop_);
  }
  else {
    if ((config->GetStreamwise_Periodic_PressureDrop() != SPPressureDrop_) && (rank == MASTER_NODE))
      cout <<"WARNING: Discarding the STREAMWISE_PERIODIC_PRESSURE_DROP in the direct solution file." << endl;
  }

  /*--- External iteration ---*/

  if ((config->GetDiscard_InFiles() == false) && (!adjoint || (adjoint && config->GetRestart())))
    config->SetExtIter_OffSet(InnerIter_);

}

void CSolver::LoadInletProfile(CGeometry **geometry,
                               CSolver ***solver,
                               CConfig *config,
                               int val_iter,
                               unsigned short val_kind_solver,
                               unsigned short val_kind_marker) const {

  /*-- First, set the solver and marker kind for the particular problem at
   hand. Note that, in the future, these routines can be used for any solver
   and potentially any marker type (beyond inlets). ---*/

  const auto KIND_SOLVER = val_kind_solver;
  const auto KIND_MARKER = val_kind_marker;

  const bool time_stepping = (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST) ||
                             (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND) ||
                             (config->GetTime_Marching() == TIME_MARCHING::TIME_STEPPING);

  const auto iZone = config->GetiZone();
  const auto nZone = config->GetnZone();

  auto profile_filename = config->GetInlet_FileName();

  const auto turbulence = config->GetKind_Turb_Model() != TURB_MODEL::NONE;
  const unsigned short nVar_Turb = turbulence ? solver[MESH_0][TURB_SOL]->GetnVar() : 0;

  const auto species = config->GetKind_Species_Model() != SPECIES_MODEL::NONE;
  const unsigned short nVar_Species = species ? solver[MESH_0][SPECIES_SOL]->GetnVar() : 0;

  /*--- names of the columns in the profile ---*/
  vector<string> columnNames;
  vector<string> columnValues;

  /*--- Count the number of columns that we have for this flow case,
   excluding the coordinates. Here, we have 2 entries for the total
   conditions or mass flow, another nDim for the direction vector, and
   finally entries for the number of turbulence variables. This is only
   necessary in case we are writing a template profile file or for Inlet
   Interpolation purposes. ---*/

  const unsigned short nCol_InletFile = 2 + nDim + nVar_Turb + nVar_Species;

  /*--- for incompressible flow, we can switch the energy equation off ---*/
  /*--- for now, we write the temperature even if we are not using it ---*/
  /*--- because a number of routines depend on the presence of the temperature field ---*/
  //if (config->GetEnergy_Equation() ==false)
  //nCol_InletFile = nCol_InletFile -1;

  /*--- Multizone problems require the number of the zone to be appended. ---*/

  if (nZone > 1)
    profile_filename = config->GetMultizone_FileName(profile_filename, iZone, ".dat");

  /*--- Modify file name for an unsteady restart ---*/

  if (time_stepping)
    profile_filename = config->GetUnsteady_FileName(profile_filename, val_iter, ".dat");


  // create vector of column names
  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- Skip if this is the wrong type of marker. ---*/
    if (config->GetMarker_All_KindBC(iMarker) != KIND_MARKER) continue;

    string Marker_Tag = config->GetMarker_All_TagBound(iMarker);
    su2double p_total   = config->GetInlet_Ptotal(Marker_Tag);
    su2double t_total   = config->GetInlet_Ttotal(Marker_Tag);
    auto flow_dir = config->GetInlet_FlowDir(Marker_Tag);
    std::stringstream columnName,columnValue;

    columnValue << setprecision(15);
    columnValue << std::scientific;

    columnValue << t_total << "\t" << p_total <<"\t";
    for (unsigned short iDim = 0; iDim < nDim; iDim++) {
      columnValue << flow_dir[iDim] <<"\t";
    }

    columnName << "# COORD-X  " << setw(24) << "COORD-Y    " << setw(24);
    if(nDim==3) columnName << "COORD-Z    " << setw(24);

    if (config->GetKind_Regime()==ENUM_REGIME::COMPRESSIBLE){
      switch (config->GetKind_Inlet()) {
        /*--- compressible conditions ---*/
        case INLET_TYPE::TOTAL_CONDITIONS:
          columnName << "TEMPERATURE" << setw(24) << "PRESSURE   " << setw(24);
          break;
        case INLET_TYPE::MASS_FLOW:
          columnName << "DENSITY    " << setw(24) << "VELOCITY   " << setw(24);
          break;
        default:
          SU2_MPI::Error("Unsupported INLET_TYPE.", CURRENT_FUNCTION);
          break;        }
    } else {
      switch (config->GetKind_Inc_Inlet(Marker_Tag)) {
        /*--- incompressible conditions ---*/
        case INLET_TYPE::VELOCITY_INLET:
          columnName << "TEMPERATURE" << setw(24) << "VELOCITY   " << setw(24);
          break;
        case INLET_TYPE::PRESSURE_INLET:
          columnName << "TEMPERATURE" << setw(24) << "PRESSURE   " << setw(24);
          break;
        default:
          SU2_MPI::Error("Unsupported INC_INLET_TYPE.", CURRENT_FUNCTION);
          break;
      }
    }

    columnName << "NORMAL-X   " << setw(24) << "NORMAL-Y   " << setw(24);
    if(nDim==3)  columnName << "NORMAL-Z   " << setw(24);

    switch (TurbModelFamily(config->GetKind_Turb_Model())) {
      case TURB_FAMILY::NONE: break;
      case TURB_FAMILY::SA:
        /*--- 1-equation turbulence model: SA ---*/
        columnName << "NU_TILDE   " << setw(24);
        columnValue << config->GetNuFactor_FreeStream() * config->GetViscosity_FreeStream() / config->GetDensity_FreeStream() <<"\t";
        break;
      case TURB_FAMILY::KW:
        /*--- 2-equation turbulence model (SST) ---*/
        columnName << "TKE        " << setw(24) << "DISSIPATION" << setw(24);
        columnValue << config->GetTke_FreeStream() << "\t" << config->GetOmega_FreeStream() <<"\t";
        break;
    }

    switch (config->GetKind_Species_Model()) {
      case SPECIES_MODEL::NONE: break;
      case SPECIES_MODEL::PASSIVE_SCALAR:
        for (unsigned short iVar = 0; iVar < nVar_Species; iVar++) {
          columnName << "SPECIES_" + std::to_string(iVar) + "  " << setw(24);
          columnValue << config->GetInlet_SpeciesVal(Marker_Tag)[iVar] << "\t";
        }
        break;
    }

    columnNames.push_back(columnName.str());
    columnValues.push_back(columnValue.str());

  }


  /*--- Read the profile data from an ASCII file. ---*/

  CMarkerProfileReaderFVM profileReader(geometry[MESH_0], config, profile_filename, KIND_MARKER, nCol_InletFile, columnNames,columnValues);

  /*--- Load data from the restart into correct containers. ---*/

  unsigned long Marker_Counter = 0;
  unsigned short local_failure = 0;

  const su2double tolerance = config->GetInlet_Profile_Matching_Tolerance();

  for (auto iMarker = 0ul; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- Skip if this is the wrong type of marker. ---*/

    if (config->GetMarker_All_KindBC(iMarker) != KIND_MARKER) continue;

    /*--- Get tag in order to identify the correct inlet data. ---*/

    const auto Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (auto jMarker = 0ul; jMarker < profileReader.GetNumberOfProfiles(); jMarker++) {

      /*--- If we have not found the matching marker string, continue to next marker. ---*/

      if (profileReader.GetTagForProfile(jMarker) != Marker_Tag) continue;

      /*--- Increment our counter for marker matches. ---*/

      Marker_Counter++;

      /*--- Get data for this profile. ---*/

      const vector<passivedouble>& Inlet_Data = profileReader.GetDataForProfile(jMarker);
      const auto nColumns = profileReader.GetNumberOfColumnsInProfile(jMarker);
      vector<su2double> Inlet_Data_Interpolated ((nCol_InletFile+nDim)*geometry[MESH_0]->nVertex[iMarker]);

      /*--- Define Inlet Values vectors before and after interpolation (if needed) ---*/
      vector<su2double> Inlet_Values(nCol_InletFile+nDim);
      vector<su2double> Inlet_Interpolated(nColumns);

      const auto nRows = profileReader.GetNumberOfRowsInProfile(jMarker);

      /*--- Pointer to call Set and Evaluate functions. ---*/
      vector<C1DInterpolation*> interpolator(nColumns,nullptr);
      string interpolation_function, interpolation_type;

      /*--- Define the reference for interpolation. ---*/
      unsigned short radius_index=0;
      vector<su2double> InletRadii = profileReader.GetColumnForProfile(jMarker, radius_index);
      vector<su2double> Interpolation_Column (nRows);

      bool Interpolate = true;

      switch(config->GetKindInletInterpolationFunction()){

        case (INLET_SPANWISE_INTERP::NONE):
          Interpolate = false;
          break;

        case (INLET_SPANWISE_INTERP::AKIMA_1D):
          for (auto iCol=0ul; iCol < nColumns; iCol++){
            Interpolation_Column = profileReader.GetColumnForProfile(jMarker, iCol);
            interpolator[iCol] = new CAkimaInterpolation(InletRadii,Interpolation_Column);
          }
          interpolation_function = "AKIMA";
          break;

        case (INLET_SPANWISE_INTERP::LINEAR_1D):
          for (auto iCol=0ul; iCol < nColumns; iCol++){
            Interpolation_Column = profileReader.GetColumnForProfile(jMarker, iCol);
            interpolator[iCol] = new CLinearInterpolation(InletRadii,Interpolation_Column);
          }
          interpolation_function = "LINEAR";
          break;

        case (INLET_SPANWISE_INTERP::CUBIC_1D):
          for (auto iCol=0ul; iCol < nColumns; iCol++){
            Interpolation_Column = profileReader.GetColumnForProfile(jMarker, iCol);
            interpolator[iCol] = new CCubicSpline(InletRadii,Interpolation_Column);
          }
          interpolation_function = "CUBIC";
          break;

        default:
          SU2_MPI::Error("Unknown type of interpolation function for inlets.\n",CURRENT_FUNCTION);
          break;
      }

      if (Interpolate){
        switch(config->GetKindInletInterpolationType()){
          case(INLET_INTERP_TYPE::VR_VTHETA):
            interpolation_type="VR_VTHETA";
            break;
          case(INLET_INTERP_TYPE::ALPHA_PHI):
            interpolation_type="ALPHA_PHI";
            break;
        }
        cout<<"Inlet Interpolation being done using "<<interpolation_function
            <<" function and type "<<interpolation_type<<" for "<< Marker_Tag<<endl;
        if(nDim == 3)
          cout<<"Ensure the flow direction is in z direction"<<endl;
        else if (nDim == 2)
          cout<<"Ensure the flow direction is in x direction"<<endl;
      }
      else {
        cout<<"No Inlet Interpolation being used"<<endl;
      }

      /*--- Loop through the nodes on this marker. ---*/

      for (auto iVertex = 0ul; iVertex < geometry[MESH_0]->nVertex[iMarker]; iVertex++) {

        const auto iPoint = geometry[MESH_0]->vertex[iMarker][iVertex]->GetNode();
        const auto Coord = geometry[MESH_0]->nodes->GetCoord(iPoint);

        if (!Interpolate) {

          su2double min_dist = 1e16;

          /*--- Find the distance to the closest point in our inlet profile data. ---*/

          for (auto iRow = 0ul; iRow < nRows; iRow++) {

            /*--- Get the coords for this data point. ---*/

            const auto index = iRow*nColumns;

            const auto dist = GeometryToolbox::Distance(nDim, Coord, &Inlet_Data[index]);

            /*--- Check is this is the closest point and store data if so. ---*/

            if (dist < min_dist) {
              min_dist = dist;
              for (auto iVar = 0ul; iVar < nColumns; iVar++)
                Inlet_Values[iVar] = Inlet_Data[index+iVar];
            }

          }

          /*--- If the diff is less than the tolerance, match the two.
          We could modify this to simply use the nearest neighbor, or
          eventually add something more elaborate here for interpolation. ---*/

          if (min_dist < tolerance) {

            solver[MESH_0][KIND_SOLVER]->SetInletAtVertex(Inlet_Values.data(), iMarker, iVertex);

          } else {

            unsigned long GlobalIndex = geometry[MESH_0]->nodes->GetGlobalIndex(iPoint);
            cout << "WARNING: Did not find a match between the points in the inlet file\n";
            cout << "and point " << GlobalIndex;
            cout << std::scientific;
            cout << " at location: [" << Coord[0] << ", " << Coord[1];
            if (nDim==3) cout << ", " << Coord[2];
            cout << "]\n";
            cout << "Distance to closest point: " << min_dist << "\n";
            cout << "Current tolerance:         " << tolerance << "\n\n";
            cout << "You can increase the tolerance for point matching by changing the value\n";
            cout << "of the option INLET_MATCHING_TOLERANCE in your *.cfg file." << endl;
            local_failure++;
            break;
          }

        }
        else { // Interpolate

          /* --- Calculating the radius and angle of the vertex ---*/
          /* --- Flow should be in z direction for 3D cases ---*/
          /* --- Or in x direction for 2D cases ---*/
          const su2double Interp_Radius = sqrt(pow(Coord[0],2)+ pow(Coord[1],2));
          const su2double Theta = atan2(Coord[1],Coord[0]);

          /* --- Evaluating and saving the final spline data ---*/
          for (auto iVar=0ul; iVar < nColumns; iVar++){

            /*---Evaluate spline will get the respective value of the Data set (column) specified
            for that interpolator[iVar], cycling through all columns to get all the
            data for that vertex ---*/
            Inlet_Interpolated[iVar]=interpolator[iVar]->EvaluateSpline(Interp_Radius);
            if (Interp_Radius < InletRadii.front() || Interp_Radius > InletRadii.back()) {
              cout << "WARNING: Did not find a match between the radius in the inlet file " ;
              cout << std::scientific;
              cout << "at location: [" << Coord[0] << ", " << Coord[1];
              if (nDim == 3) {cout << ", " << Coord[2];}
              cout << "]";
              cout << " with Radius: "<< Interp_Radius << endl;
              cout << "You can add a row for Radius: " << Interp_Radius <<" in the inlet file ";
              cout << "to eliminate this issue or give proper data" << endl;
              local_failure++;
              break;
            }
          }

          /*--- Correcting for Interpolation Type ---*/

          Inlet_Values = CorrectedInletValues(Inlet_Interpolated, Theta, nDim, Coord,
                                              nVar_Turb, config->GetKindInletInterpolationType());

          solver[MESH_0][KIND_SOLVER]->SetInletAtVertex(Inlet_Values.data(), iMarker, iVertex);

          for (unsigned short iVar=0; iVar < (nCol_InletFile+nDim); iVar++)
            Inlet_Data_Interpolated[iVertex*(nCol_InletFile+nDim)+iVar] = Inlet_Values[iVar];

        }

      } // end iVertex loop

      if (config->GetPrintInlet_InterpolatedData()) {
        PrintInletInterpolatedData(Inlet_Data_Interpolated, profileReader.GetTagForProfile(jMarker),
                                   geometry[MESH_0]->nVertex[iMarker], nDim, nCol_InletFile+nDim);
      }

      for (auto& interp : interpolator) delete interp;

    } // end jMarker loop

    if (local_failure > 0) break;

  } // end iMarker loop

  unsigned short global_failure;
  SU2_MPI::Allreduce(&local_failure, &global_failure, 1, MPI_UNSIGNED_SHORT, MPI_SUM, SU2_MPI::GetComm());

  if (global_failure > 0) {
    SU2_MPI::Error("Prescribed inlet data does not match markers within tolerance.", CURRENT_FUNCTION);
  }

  /*--- Copy the inlet data down to the coarse levels if multigrid is active.
   Here, we use a face area-averaging to restrict the values. ---*/

  for (auto iMesh = 1u; iMesh <= config->GetnMGLevels(); iMesh++) {
    for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {
      if (config->GetMarker_All_KindBC(iMarker) == KIND_MARKER) {

        const auto Marker_Tag = config->GetMarker_All_TagBound(iMarker);

        /* Check the number of columns and allocate temp array. */

        unsigned short nColumns = 0;
        for (auto jMarker = 0ul; jMarker < profileReader.GetNumberOfProfiles(); jMarker++) {
          if (profileReader.GetTagForProfile(jMarker) == Marker_Tag) {
            nColumns = profileReader.GetNumberOfColumnsInProfile(jMarker);
            break;
          }
        }
        vector<su2double> Inlet_Values(nColumns);
        vector<su2double> Inlet_Fine(nColumns);

        /*--- Loop through the nodes on this marker. ---*/

        for (auto iVertex = 0ul; iVertex < geometry[iMesh]->nVertex[iMarker]; iVertex++) {

          /*--- Get the coarse mesh point and compute the boundary area. ---*/

          const auto iPoint = geometry[iMesh]->vertex[iMarker][iVertex]->GetNode();
          const auto Normal = geometry[iMesh]->vertex[iMarker][iVertex]->GetNormal();
          const su2double Area_Parent = GeometryToolbox::Norm(nDim, Normal);

          /*--- Reset the values for the coarse point. ---*/

          for (auto& v : Inlet_Values) v = 0.0;

          /*-- Loop through the children and extract the inlet values
           from those nodes that lie on the boundary as well as their
           boundary area. We build a face area-averaged value for the
           coarse point values from the fine grid points. Note that
           children from the interior volume will not be included in
           the averaging. ---*/

          for (auto iChildren = 0u; iChildren < geometry[iMesh]->nodes->GetnChildren_CV(iPoint); iChildren++) {
            const auto Point_Fine = geometry[iMesh]->nodes->GetChildren_CV(iPoint, iChildren);

            auto Area_Children = solver[iMesh-1][KIND_SOLVER]->GetInletAtVertex(Inlet_Fine.data(), Point_Fine, KIND_MARKER,
                                                                                Marker_Tag, geometry[iMesh-1], config);
            for (auto iVar = 0u; iVar < nColumns; iVar++)
              Inlet_Values[iVar] += Inlet_Fine[iVar]*Area_Children/Area_Parent;
          }

          /*--- Set the boundary area-averaged inlet values for the coarse point. ---*/

          solver[iMesh][KIND_SOLVER]->SetInletAtVertex(Inlet_Values.data(), iMarker, iVertex);

        }
      }
    }
  }

}


void CSolver::ComputeVertexTractions(CGeometry *geometry, const CConfig *config){

  const bool viscous_flow = config->GetViscous();
  const su2double Pressure_Inf = config->GetPressure_FreeStreamND();

  for (auto iMarker = 0u; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- If this is defined as a wall ---*/
    if (!config->GetSolid_Wall(iMarker)) continue;

    // Loop over the vertices
    for (auto iVertex = 0ul; iVertex < geometry->nVertex[iMarker]; iVertex++) {

      // Recover the point index
      const auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

      su2double auxForce[3] = {0.0};

      // Check if the node belongs to the domain (i.e, not a halo node).
      if (geometry->nodes->GetDomain(iPoint)) {

        // Get the normal at the vertex: this normal goes inside the fluid domain.
        const su2double* Normal = geometry->vertex[iMarker][iVertex]->GetNormal();

        // Retrieve the values of pressure
        const su2double Pn = base_nodes->GetPressure(iPoint);

        // Calculate tn in the fluid nodes for the inviscid term --> Units of force (non-dimensional).
        for (unsigned short iDim = 0; iDim < nDim; iDim++)
          auxForce[iDim] = -(Pn-Pressure_Inf)*Normal[iDim];

        // Calculate tn in the fluid nodes for the viscous term
        if (viscous_flow) {
          const su2double Viscosity = base_nodes->GetLaminarViscosity(iPoint);
          su2double Tau[3][3];
          CNumerics::ComputeStressTensor(nDim, Tau, base_nodes->GetVelocityGradient(iPoint), Viscosity);
          for (unsigned short iDim = 0; iDim < nDim; iDim++) {
            auxForce[iDim] += GeometryToolbox::DotProduct(nDim, Tau[iDim], Normal);
          }
        }
      }

      // Redimensionalize the forces (Lref is 1, thus only Pref is needed).
      for (unsigned short iDim = 0; iDim < nDim; iDim++) {
        VertexTraction[iMarker][iVertex][iDim] = config->GetPressure_Ref() * auxForce[iDim];
      }
    }
  }

}

void CSolver::RegisterVertexTractions(CGeometry *geometry, const CConfig *config){

  unsigned short iMarker, iDim;
  unsigned long iVertex, iPoint;

  /*--- Loop over all the markers ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- If this is defined as a wall ---*/
    if (!config->GetSolid_Wall(iMarker)) continue;

    /*--- Loop over the vertices ---*/
    SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
    for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

      /*--- Recover the point index ---*/
      iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

      /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
      if (!geometry->nodes->GetDomain(iPoint)) continue;

      /*--- Register the vertex traction as output ---*/
      for (iDim = 0; iDim < nDim; iDim++) {
        AD::RegisterOutput(VertexTraction[iMarker][iVertex][iDim]);
      }
    }
    END_SU2_OMP_FOR
  }

}

void CSolver::SetVertexTractionsAdjoint(CGeometry *geometry, const CConfig *config){

  unsigned short iMarker, iDim;
  unsigned long iVertex, iPoint;

  /*--- Loop over all the markers ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    /*--- If this is defined as a wall ---*/
    if (!config->GetSolid_Wall(iMarker)) continue;

    /*--- Loop over the vertices ---*/
    SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
    for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

      /*--- Recover the point index ---*/
      iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

      /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
      if (!geometry->nodes->GetDomain(iPoint)) continue;

      /*--- Set the adjoint of the vertex traction from the value received ---*/
      for (iDim = 0; iDim < nDim; iDim++) {
        SU2_TYPE::SetDerivative(VertexTraction[iMarker][iVertex][iDim],
                                SU2_TYPE::GetValue(VertexTractionAdjoint[iMarker][iVertex][iDim]));
      }
    }
    END_SU2_OMP_FOR
  }

}


void CSolver::SetVerificationSolution(unsigned short nDim,
                                      unsigned short nVar,
                                      CConfig        *config) {

  /*--- Determine the verification solution to be set and
        allocate memory for the corresponding class. ---*/
  switch( config->GetVerification_Solution() ) {

    case VERIFICATION_SOLUTION::NONE:
      VerificationSolution = nullptr; break;
    case VERIFICATION_SOLUTION::INVISCID_VORTEX:
      VerificationSolution = new CInviscidVortexSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::RINGLEB:
      VerificationSolution = new CRinglebSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::NS_UNIT_QUAD:
      VerificationSolution = new CNSUnitQuadSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::TAYLOR_GREEN_VORTEX:
      VerificationSolution = new CTGVSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::INC_TAYLOR_GREEN_VORTEX:
      VerificationSolution = new CIncTGVSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_NS_UNIT_QUAD:
      VerificationSolution = new CMMSNSUnitQuadSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_NS_UNIT_QUAD_WALL_BC:
      VerificationSolution = new CMMSNSUnitQuadSolutionWallBC(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_NS_TWO_HALF_CIRCLES:
      VerificationSolution = new CMMSNSTwoHalfCirclesSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_NS_TWO_HALF_SPHERES:
      VerificationSolution = new CMMSNSTwoHalfSpheresSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_INC_EULER:
      VerificationSolution = new CMMSIncEulerSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::MMS_INC_NS:
      VerificationSolution = new CMMSIncNSSolution(nDim, nVar, MGLevel, config); break;
    case VERIFICATION_SOLUTION::USER_DEFINED_SOLUTION:
      VerificationSolution = new CUserDefinedSolution(nDim, nVar, MGLevel, config); break;
  }
}

void CSolver::ComputeResidual_Multizone(const CGeometry *geometry, const CConfig *config){

  SU2_OMP_PARALLEL {

  /*--- Set Residuals to zero ---*/
  SU2_OMP_MASTER
  for (unsigned short iVar = 0; iVar < nVar; iVar++){
    Residual_BGS[iVar] = 0.0;
    Residual_Max_BGS[iVar] = 0.0;
  }
  END_SU2_OMP_MASTER

  vector<su2double> resMax(nVar,0.0), resRMS(nVar,0.0);
  vector<const su2double*> coordMax(nVar,nullptr);
  vector<unsigned long> idxMax(nVar,0);

  /*--- Set the residuals and BGSSolution_k to solution for next multizone outer iteration. ---*/
  SU2_OMP_FOR_STAT(roundUpDiv(nPoint,2*omp_get_num_threads()))
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {
    const su2double domain = (iPoint < nPointDomain);
    for (unsigned short iVar = 0; iVar < nVar; iVar++) {
      const su2double Res = (base_nodes->Get_BGSSolution(iPoint,iVar) - base_nodes->Get_BGSSolution_k(iPoint,iVar))*domain;

      /*--- Update residual information for current thread. ---*/
      resRMS[iVar] += Res*Res;
      if (fabs(Res) > resMax[iVar]) {
        resMax[iVar] = fabs(Res);
        idxMax[iVar] = iPoint;
        coordMax[iVar] = geometry->nodes->GetCoord(iPoint);
      }
    }
  }
  END_SU2_OMP_FOR

  /*--- Reduce residual information over all threads in this rank. ---*/
  SU2_OMP_CRITICAL
  for (unsigned short iVar = 0; iVar < nVar; iVar++) {
    Residual_BGS[iVar] += resRMS[iVar];
    AddRes_Max_BGS(iVar, resMax[iVar], geometry->nodes->GetGlobalIndex(idxMax[iVar]), coordMax[iVar]);
  }
  END_SU2_OMP_CRITICAL
  SU2_OMP_BARRIER

  SetResidual_BGS(geometry, config);

  }
  END_SU2_OMP_PARALLEL
}

void CSolver::HB_Operator(CConfig *config, vector<vector<su2double>>& D, vector<vector<su2double>>& E, int nInstHB, su2double Omega) {

   int NT = nInstHB, NH = (NT-1)/2, i, j, k; 
   su2double a = 2*PI_NUMBER/NT;

   //vector<vector<su2double>> E(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> Einv(NT,vector<su2double>(NT,0.0));
   //vector<vector<su2double>> D(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> A(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> tmp(NT,vector<su2double>(NT,0.0));


   for (j=0;j<NT;j++) {
	  E[0][j] = 1.0/NT;
   } 
   for (i=0;i<NH;i++) {
   for (j=0;j<NT;j++) {
	  E[2*i+1][j] = 2*cos(j*(i+1)*a)/NT;
   }
   }
   for (i=0;i<NH;i++) {
   for (j=0;j<NT;j++) {
	  E[2*i+2][j] = 2*sin(j*(i+1)*a)/NT;
   }
   }

   for (j=0;j<NT;j++) {
	  Einv[j][0] = 1;
   } 
   for (j=0;j<NH;j++) {
   for (i=0;i<NT;i++) {
	  Einv[i][2*j+1] = cos(i*(j+1)*a);
   }
   }
   for (j=0;j<NH;j++) {
   for (i=0;i<NT;i++) {
	  Einv[i][2*j+2] = sin(i*(j+1)*a);
   }
   }

   for (i=0;i<NH;i++) {

         A[2*i+1][2*i+2] = -(i+1);
	 
         A[2*i+2][2*i+1] =  (i+1);
         
   }

   for (i=0;i<NT;i++) {
   for (j=0;j<NT;j++) {	
   for (k=0;k<NT;k++) {

	  tmp[i][j] += A[i][k] * E[k][j];
   
   }
   }
   }
   for (i=0;i<NT;i++) {
   for (j=0;j<NT;j++) {	
   for (k=0;k<NT;k++) {

	  D[i][j] += Einv[i][k] * tmp[k][j];
   
   }
   }
   }
   for (i=0;i<NT;i++) {
   for (j=0;j<NT;j++) {	
  	  D[i][j] = -D[i][j];
    
   }
   }

}

void CSolver::HB_Operator_Complex(CConfig *config, vector<vector<su2double>>& D, vector<vector<su2double>>& E, int nInstHB, su2double Omega) {

   int NT = nInstHB, NH = (NT-1)/2, i, j, k; 
   su2double a = 2*PI_NUMBER/NT;

   //vector<vector<su2double>> E(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> Einv(NT,vector<su2double>(NT,0.0));
   //vector<vector<su2double>> D(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> A(NT,vector<su2double>(NT,0.0));
   vector<vector<su2double>> tmp(NT,vector<su2double>(NT,0.0));


   for (j=0;j<NT;j++) {
	  E[0][j] = 1.0/NT;
   } 
   for (i=0;i<NH;i++) {
   for (j=0;j<NT;j++) {
	  E[2*i+1][j] = 2*cos(j*(i+1)*a)/NT;
   }
   }
   for (i=0;i<NH;i++) {
   for (j=0;j<NT;j++) {
	  E[2*i+2][j] = 2*sin(j*(i+1)*a)/NT;
   }
   }

   for (j=0;j<NT;j++) {
	  Einv[j][0] = 1;
   } 
   for (j=0;j<NH;j++) {
   for (i=0;i<NT;i++) {
	  Einv[i][2*j+1] = cos(i*(j+1)*a);
   }
   }
   for (j=0;j<NH;j++) {
   for (i=0;i<NT;i++) {
	  Einv[i][2*j+2] = sin(i*(j+1)*a);
   }
   }

   for (i=0;i<NH;i++) {

         A[2*i+1][2*i+2] = -(i+1);
	 
         A[2*i+2][2*i+1] =  (i+1);
         
   }

   for (i=0;i<NT;i++) {
   for (j=0;j<NT;j++) {	
   for (k=0;k<NT;k++) {

	  tmp[i][j] += A[i][k] * E[k][j];
   
   }
   }
   }
   for (i=0;i<NT;i++) {
   for (j=0;j<NT;j++) {	

	  D[i][j] = - tmp[i][j];
   
   }
   }

}


void CSolver::BasicLoadRestart(CGeometry *geometry, const CConfig *config, const string& filename, unsigned long skipVars) {

  /*--- Read and store the restart metadata. ---*/

//  Read_SU2_Restart_Metadata(geometry[MESH_0], config, true, filename);

  /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

  if (config->GetRead_Binary_Restart()) {
    Read_SU2_Restart_Binary(geometry, config, filename);
  } else {
    Read_SU2_Restart_ASCII(geometry, config, filename);
  }

  /*--- Load data from the restart into correct containers. ---*/

  unsigned long iPoint_Global_Local = 0;

  for (auto iPoint_Global = 0ul; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++ ) {

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    const auto iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local > -1) {

      /*--- We need to store this point's data, so jump to the correct
       offset in the buffer of data from the restart file and load it. ---*/

      const auto index = iPoint_Global_Local*Restart_Vars[1] + skipVars;

      for (auto iVar = 0u; iVar < nVar; iVar++) {
        base_nodes->SetSolution(iPoint_Local, iVar, Restart_Data[index+iVar]);
      }

      iPoint_Global_Local++;
    }

  }

  /*--- Delete the class memory that is used to load the restart. ---*/

  delete [] Restart_Vars;  Restart_Vars = nullptr;
  delete [] Restart_Data;  Restart_Data = nullptr;

  /*--- Detect a wrong solution file ---*/

  if (iPoint_Global_Local != nPointDomain) {
    SU2_MPI::Error(string("The solution file ") + filename + string(" doesn't match with the mesh file!\n") +
                   string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
  }
}

void CSolver::SavelibROM(CGeometry *geometry, CConfig *config, bool converged) {

#if defined(HAVE_LIBROM) && !defined(CODI_FORWARD_TYPE) && !defined(CODI_REVERSE_TYPE)
  const bool unsteady            = config->GetTime_Domain();
  const string filename          = config->GetlibROMbase_FileName();
  const unsigned long TimeIter   = config->GetTimeIter();
  const unsigned long nTimeIter  = config->GetnTime_Iter();
  const int maxBasisDim          = config->GetMax_BasisDim();
  const int save_freq            = config->GetRom_SaveFreq();
  int dim = int(nPointDomain * nVar);
  bool incremental = false;

  if (!u_basis_generator) {

    /*--- Define SVD basis generator ---*/
    auto timesteps = static_cast<int>(nTimeIter - TimeIter);
    CAROM::Options svd_options = CAROM::Options(dim, timesteps, -1,
                                                false, true).setMaxBasisDimension(int(maxBasisDim));

    if (config->GetKind_PODBasis() == POD_KIND::STATIC) {
      if (rank == MASTER_NODE) std::cout << "Creating static basis generator." << std::endl;

      if (unsteady) {
        if (rank == MASTER_NODE) std::cout << "Incremental basis generator recommended for unsteady simulations." << std::endl;
      }
    }
    else {
      if (rank == MASTER_NODE) std::cout << "Creating incremental basis generator." << std::endl;

      svd_options.setIncrementalSVD(1.0e-3, config->GetDelta_UnstTime(),
                                    1.0e-2, config->GetDelta_UnstTime()*nTimeIter, true).setDebugMode(false);
      incremental = true;
    }

    u_basis_generator.reset(new CAROM::BasisGenerator(
      svd_options, incremental,
      filename));

    // Save mesh ordering
    std::ofstream f;
    f.open(filename + "_mesh_" + to_string(rank) + ".csv");
      for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {
        unsigned long globalPoint = geometry->nodes->GetGlobalIndex(iPoint);
        auto Coord = geometry->nodes->GetCoord(iPoint);

        for (unsigned long iDim; iDim < nDim; iDim++) {
          f << Coord[iDim] << ", ";
        }
        f << globalPoint << "\n";
      }
    f.close();
  }

  if (unsteady && (TimeIter % save_freq == 0)) {
    // give solution and time steps to libROM:
    su2double dt = config->GetDelta_UnstTime();
    su2double t =  config->GetCurrent_UnstTime();
    u_basis_generator->takeSample(const_cast<su2double*>(base_nodes->GetSolution().data()), t, dt);
  }

  /*--- End collection of data and save POD ---*/

  if (converged) {

    if (!unsteady) {
       // dt is different for each node, so just use a placeholder dt
       su2double dt = base_nodes->GetDelta_Time(0);
       su2double t = dt*TimeIter;
       u_basis_generator->takeSample(const_cast<su2double*>(base_nodes->GetSolution().data()), t, dt);
    }

    if (config->GetKind_PODBasis() == POD_KIND::STATIC) {
      u_basis_generator->writeSnapshot();
    }

    if (rank == MASTER_NODE) std::cout << "Computing SVD" << std::endl;
    int rom_dim = u_basis_generator->getSpatialBasis()->numColumns();

    if (rank == MASTER_NODE) std::cout << "Basis dimension: " << rom_dim << std::endl;
    u_basis_generator->endSamples();

    if (rank == MASTER_NODE) std::cout << "ROM Sampling ended" << std::endl;
  }

#else
  SU2_MPI::Error("SU2 was not compiled with libROM support.", CURRENT_FUNCTION);
#endif

}
