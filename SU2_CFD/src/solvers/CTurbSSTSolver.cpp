/*!
 * \file CTurbSSTSolver.cpp
 * \brief Main subrotuines of CTurbSSTSolver class
 * \author F. Palacios, A. Bueno
 * \version 7.0.3 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
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

#include "../../include/solvers/CTurbSSTSolver.hpp"
#include "../../include/variables/CTurbSSTVariable.hpp"
#include "../../include/gradients/computeGradientsGreenGauss.hpp"
#include "../../include/gradients/computeGradientsLeastSquares.hpp"
#include "../../include/limiters/computeLimiters.hpp"
#include "../../../Common/include/omp_structure.hpp"

CTurbSSTSolver::CTurbSSTSolver(void) : CTurbSolver() { }

CTurbSSTSolver::CTurbSSTSolver(CGeometry *geometry, CConfig *config, unsigned short iMesh)
    : CTurbSolver(geometry, config) {
  unsigned short iVar, iDim, nLineLets;
  unsigned long iPoint;
  ifstream restart_file;
  string text_line;

  bool multizone = config->GetMultizone_Problem();

  /*--- Array initialization ---*/

  Gamma = config->GetGamma();
  Gamma_Minus_One = Gamma - 1.0;

  /*--- Dimension of the problem --> dependent on the turbulence model. ---*/

  nVar = 2;
  nPrimVar = 2;
  nPoint = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();

  /*--- Initialize nVarGrad for deallocation ---*/

  nVarGrad = nVar;

  /*--- Define geometry constants in the solver structure ---*/

  nDim = geometry->GetnDim();

  /*--- Single grid simulation ---*/

  if (iMesh == MESH_0) {

    /*--- Define some auxiliary vector related with the residual ---*/

    Residual = new su2double[nVar]();
    Residual_RMS = new su2double[nVar]();
    Residual_Ini = new su2double[nVar]();
    Residual_i = new su2double[nVar]();
    Residual_j = new su2double[nVar]();
    Residual_Max = new su2double[nVar]();

    /*--- Define some structures for locating max residuals ---*/

    Point_Max = new unsigned long[nVar]();
    Point_Max_Coord = new su2double*[nVar];
    for (iVar = 0; iVar < nVar; iVar++) {
      Point_Max_Coord[iVar] = new su2double[nDim]();
    }

    /*--- Define some auxiliary vector related with the solution ---*/

    Solution = new su2double[nVar];
    Solution_i = new su2double[nVar]; Solution_j = new su2double[nVar];
    Primitive = new su2double[nVar];
    Primitive_i = new su2double[nVar]; Primitive_j = new su2double[nVar];

    /*--- Jacobians and vector structures for implicit computations ---*/

    Jacobian_i = new su2double* [nVar];
    Jacobian_j = new su2double* [nVar];
    for (iVar = 0; iVar < nVar; iVar++) {
      Jacobian_i[iVar] = new su2double [nVar];
      Jacobian_j[iVar] = new su2double [nVar];
    }

    /*--- Initialization of the structure of the whole Jacobian ---*/

    if (rank == MASTER_NODE) cout << "Initialize Jacobian structure (SST model)." << endl;
    Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, true, geometry, config, ReducerStrategy);

    if (config->GetKind_Linear_Solver_Prec() == LINELET) {
      nLineLets = Jacobian.BuildLineletPreconditioner(geometry, config);
      if (rank == MASTER_NODE) cout << "Compute linelet structure. " << nLineLets << " elements in each line (average)." << endl;
    }

    LinSysSol.Initialize(nPoint, nPointDomain, nVar, 0.0);
    LinSysRes.Initialize(nPoint, nPointDomain, nVar, 0.0);

    if (ReducerStrategy)
      EdgeFluxes.Initialize(geometry->GetnEdge(), geometry->GetnEdge(), nVar, nullptr);

    /*--- Initialize the BGS residuals in multizone problems. ---*/
    if (multizone){
      Residual_BGS = new su2double[nVar]();
      Residual_Max_BGS = new su2double[nVar]();

      /*--- Define some structures for locating max residuals ---*/

      Point_Max_BGS = new unsigned long[nVar]();
      Point_Max_Coord_BGS = new su2double*[nVar];
      for (iVar = 0; iVar < nVar; iVar++) {
        Point_Max_Coord_BGS[iVar] = new su2double[nDim]();
      }
    }

  }

  /*--- Computation of gradients by least squares ---*/

  if (config->GetLeastSquaresRequired()) {
    /*--- S matrix := inv(R)*traspose(inv(R)) ---*/
    Smatrix = new su2double* [nDim];
    for (iDim = 0; iDim < nDim; iDim++)
      Smatrix[iDim] = new su2double [nDim];
    /*--- c vector := transpose(WA)*(Wb) ---*/
    Cvector = new su2double* [nVar];
    for (iVar = 0; iVar < nVar; iVar++)
      Cvector[iVar] = new su2double [nDim];
  }

  /*--- Initialize value for model constants ---*/
  constants[0] = 0.85;   //sigma_k1
  constants[1] = 1.0;    //sigma_k2
  constants[2] = 0.5;    //sigma_om1
  constants[3] = 0.856;  //sigma_om2
  constants[4] = 0.075;  //beta_1
  constants[5] = 0.0828; //beta_2
  constants[6] = 0.09;   //betaStar
  constants[7] = 0.31;   //a1
  constants[8] = constants[4]/constants[6] - constants[2]*0.41*0.41/sqrt(constants[6]);  //alfa_1
  constants[9] = constants[5]/constants[6] - constants[3]*0.41*0.41/sqrt(constants[6]);  //alfa_2

  /*--- Far-field flow state quantities and initialization. ---*/

  kine_Inf  = config->GetTke_FreeStreamND();
  omega_Inf = config->GetOmega_FreeStreamND();

  const su2double rho_Inf = config->GetDensity_FreeStreamND();
  const su2double muT_Inf = rho_Inf*kine_Inf/omega_Inf;

  /*--- Initialize lower and upper limits---*/

  lowerlimit[0] = 1.0e-16;
  upperlimit[0] = 1.0e15;

  lowerlimit[1] = 1.0e-16;
  upperlimit[1] = 1.0e15;
  

  /*--- Initialize the solution to the far-field state everywhere. ---*/

  nodes = new CTurbSSTVariable(kine_Inf, omega_Inf, muT_Inf, nPoint, nDim, nVar, constants, config);
  SetBaseClassPointerToNodes();

  /*--- MPI solution ---*/

  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

  /*--- Initializate quantities for SlidingMesh Interface ---*/

  unsigned long iMarker;

  SlidingState       = new su2double*** [nMarker]();
  SlidingStateNodes  = new int*         [nMarker]();

  for (iMarker = 0; iMarker < nMarker; iMarker++){

    if (config->GetMarker_All_KindBC(iMarker) == FLUID_INTERFACE){

      SlidingState[iMarker]       = new su2double**[geometry->GetnVertex(iMarker)]();
      SlidingStateNodes[iMarker]  = new int        [geometry->GetnVertex(iMarker)]();

      for (iPoint = 0; iPoint < geometry->GetnVertex(iMarker); iPoint++)
        SlidingState[iMarker][iPoint] = new su2double*[nPrimVar+1]();
    }
  }

  /*-- Allocation of inlets has to happen in derived classes (not CTurbSolver),
    due to arbitrary number of turbulence variables ---*/

  Inlet_TurbVars = new su2double**[nMarker];
  for (unsigned long iMarker = 0; iMarker < nMarker; iMarker++) {
    Inlet_TurbVars[iMarker] = new su2double*[nVertex[iMarker]];
    for(unsigned long iVertex=0; iVertex < nVertex[iMarker]; iVertex++){
      Inlet_TurbVars[iMarker][iVertex] = new su2double[nVar];
      Inlet_TurbVars[iMarker][iVertex][0] = kine_Inf;
      Inlet_TurbVars[iMarker][iVertex][1] = omega_Inf;
    }
  }

  /*--- The turbulence models are always solved implicitly, so set the
  implicit flag in case we have periodic BCs. ---*/

  SetImplicitPeriodic(true);

  /*--- Store the initial CFL number for all grid points. ---*/

  const su2double CFL = config->GetCFL(MGLevel)*config->GetCFLRedCoeff_Turb();
  for (iPoint = 0; iPoint < nPoint; iPoint++) {
    nodes->SetLocalCFL(iPoint, CFL);
  }
  Min_CFL_Local = CFL;
  Max_CFL_Local = CFL;
  Avg_CFL_Local = CFL;

  /*--- Add the solver name (max 8 characters) ---*/
  SolverName = "K-W SST";

}

CTurbSSTSolver::~CTurbSSTSolver(void) {

  unsigned long iMarker, iVertex;
  unsigned short iVar;
  
  if (Primitive != NULL)   delete [] Primitive;
  if (Primitive_i != NULL) delete [] Primitive_i;
  if (Primitive_j != NULL) delete [] Primitive_j;

  if ( SlidingState != NULL ) {
    for (iMarker = 0; iMarker < nMarker; iMarker++) {
      if ( SlidingState[iMarker] != NULL ) {
        for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++)
          if ( SlidingState[iMarker][iVertex] != NULL ){
            for (iVar = 0; iVar < nPrimVar+1; iVar++)
              delete [] SlidingState[iMarker][iVertex][iVar];
            delete [] SlidingState[iMarker][iVertex];
          }
        delete [] SlidingState[iMarker];
      }
    }
    delete [] SlidingState;
  }

  if ( SlidingStateNodes != NULL ){
    for (iMarker = 0; iMarker < nMarker; iMarker++){
        if (SlidingStateNodes[iMarker] != NULL)
            delete [] SlidingStateNodes[iMarker];
    }
    delete [] SlidingStateNodes;
  }

}


void CTurbSSTSolver::Preprocessing(CGeometry *geometry, CSolver **solver, CConfig *config,
         unsigned short iMesh, unsigned short iRKStep, unsigned short RunTime_EqSystem, bool Output) {
  
  /*--- Clear residual and system matrix, not needed for
   * reducer strategy as we write over the entire matrix. ---*/
  if (!ReducerStrategy) {
    LinSysRes.SetValZero();
    Jacobian.SetValZero();
  }
  
  /*--- Compute primitives and gradients ---*/

  Postprocessing(geometry, solver, config, iMesh);

}

void CTurbSSTSolver::Postprocessing(CGeometry *geometry, CSolver **solver, CConfig *config, unsigned short iMesh) {

  const bool limiter_turb = (config->GetKind_SlopeLimit_Turb() != NO_LIMITER) &&
                            (!config->GetEdgeLimiter_Turb()) &&
                            (config->GetInnerIter() <= config->GetLimiterIter());

  /*--- BCM: Reset non-physical ---*/
                            
  for (auto iPoint = 0ul; iPoint < nPoint; iPoint++) {
    nodes->SetNon_Physical(iPoint, false);
    // for (auto iVar = 0; iVar < nVar; iVar++) {
    //   const su2double sol = max(min(nodes->GetSolution(iPoint, iVar), upperlimit[iVar]), lowerlimit[iVar]);
    //   nodes->SetSolution(iPoint, iVar, sol);
    // }
  }
  
  SetPrimitive_Variables(solver);
  
  /*--- Compute turbulence gradients and limiters ---*/

  if (config->GetKind_Gradient_Method() == GREEN_GAUSS) SetPrimitive_Gradient_GG(geometry, config);
  if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) SetPrimitive_Gradient_LS(geometry, config);

  if (config->GetReconstructionGradientRequired()) {
    switch (config->GetKind_Gradient_Method_Recon()) {
      case GREEN_GAUSS:
        SetPrimitive_Gradient_GG(geometry, config, true); break;
      case LEAST_SQUARES:
      case WEIGHTED_LEAST_SQUARES:
        SetPrimitive_Gradient_LS(geometry, config, true); break;
      default: break;
    }
  }

  if (limiter_turb) SetPrimitive_Limiter(geometry, config);
  
  /*--- Compute eddy viscosity ---*/

  SetEddyViscosity(geometry, solver);

}

void CTurbSSTSolver::SetPrimitive_Variables(CSolver **solver) {
  
  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();

  for (auto iPoint = 0ul; iPoint < nPoint; iPoint++)
    for (auto iVar = 0u; iVar < nVar; iVar++)
      nodes->SetPrimitive(iPoint, iVar, nodes->GetSolution(iPoint, iVar)/flowNodes->GetDensity(iPoint));

}

void CTurbSSTSolver::SetEddyViscosity(CGeometry *geometry, CSolver **solver) {
  
  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();
  
  const su2double a1 = constants[7];
  
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint ++) {

    /*--- Compute blending functions ---*/
    
    const su2double rho = flowNodes->GetDensity(iPoint);
    const su2double mu  = flowNodes->GetLaminarViscosity(iPoint);
    const su2double dist = geometry->node[iPoint]->GetWall_Distance();
        
    nodes->SetBlendingFunc(iPoint, mu, dist, rho);

    /*--- Compute the eddy viscosity ---*/

    const su2double F2 = nodes->GetF2blending(iPoint);
    const su2double VorticityMag = flowNodes->GetVorticityMag(iPoint);

    const su2double kine  = nodes->GetPrimitive(iPoint,0);
    const su2double omega = nodes->GetPrimitive(iPoint,1);
    const su2double zeta  = max(omega, VorticityMag*F2/a1);
    const su2double muT   = rho*kine/zeta;

    nodes->SetmuT(iPoint,muT);
    flowNodes->SetEddyViscosity(iPoint,muT);
  }

}


//--- This is hacky, fix later
void CTurbSSTSolver::SetPrimitive_Gradient_GG(CGeometry *geometry, CConfig *config, bool reconstruction) {

  const auto& primitives = nodes->GetPrimitive();
  auto& gradient = reconstruction? nodes->GetGradient_Reconstruction() : nodes->GetGradient();
  auto kindComms = reconstruction? SOLUTION_GRADIENT_RECON : SOLUTION_GRADIENT;

  computeGradientsGreenGauss(this, kindComms, PERIODIC_SOL_GG, *geometry,
                             *config, primitives, 0, nVar, gradient);
}

void CTurbSSTSolver::SetPrimitive_Gradient_LS(CGeometry *geometry, CConfig *config, bool reconstruction) {

  /*--- Set a flag for unweighted or weighted least-squares. ---*/
  bool weighted = reconstruction? (config->GetKind_Gradient_Method_Recon() == WEIGHTED_LEAST_SQUARES)
                                : (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES);

  const auto& primitives = nodes->GetPrimitive();
  auto& rmatrix = nodes->GetRmatrix();
  auto& gradient = reconstruction? nodes->GetGradient_Reconstruction() : nodes->GetGradient();
  auto kindComms = reconstruction? SOLUTION_GRADIENT_RECON : SOLUTION_GRADIENT;
  PERIODIC_QUANTITIES kindPeriodicComm = weighted? PERIODIC_SOL_LS : PERIODIC_SOL_ULS;

  auto& smatrix = reconstruction? nodes->GetSmatrix_Reconstruction() : nodes->GetSmatrix();
  auto kindSmatComms = reconstruction? SMATRIX_RECON : SMATRIX;

  computeGradientsLeastSquares(this, kindComms, kindPeriodicComm, kindSmatComms, *geometry, *config,
                               weighted, primitives, 0, nVar, gradient, rmatrix, smatrix);
}

void CTurbSSTSolver::SetPrimitive_Limiter(CGeometry *geometry, CConfig *config) {

  auto kindLimiter = static_cast<ENUM_LIMITER>(config->GetKind_SlopeLimit_Turb());
  const auto& primitives = nodes->GetPrimitive();
  const auto& gradient = nodes->GetGradient_Reconstruction();
  auto& primMin = nodes->GetSolution_Min();
  auto& primMax = nodes->GetSolution_Max();
  auto& limiter = nodes->GetLimiter();

  computeLimiters(kindLimiter, this, SOLUTION_LIMITER, PERIODIC_LIM_SOL_1, PERIODIC_LIM_SOL_2,
            *geometry, *config, 0, nVar, primitives, gradient, primMin, primMax, limiter);
}
//--- End hacky bit

void CTurbSSTSolver::Source_Residual(CGeometry *geometry, CSolver **solver,
                                     CNumerics **numerics_container, CConfig *config, unsigned short iMesh) {

  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();

  /*--- Pick one numerics object per thread. ---*/
  CNumerics* numerics = numerics_container[SOURCE_FIRST_TERM + omp_get_thread_num()*MAX_TERMS];

  /*--- Loop over all points. ---*/

  SU2_OMP_FOR_DYN(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPointDomain; iPoint++) {

    if (geometry->node[iPoint]->GetWall_Distance() > 1.0e-10) {

      /*--- Conservative variables w/o reconstruction ---*/

      numerics->SetPrimitive(flowNodes->GetPrimitive(iPoint), nullptr);

      /*--- Gradient of the primitive and conservative variables ---*/

      numerics->SetPrimVarGradient(flowNodes->GetGradient_Primitive(iPoint), nullptr);
      numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nullptr);

      /*--- Turbulent variables w/o reconstruction ---*/

      numerics->SetTurbVar(nodes->GetPrimitive(iPoint), nullptr);

      /*--- Set volume ---*/

      numerics->SetVolume(geometry->node[iPoint]->GetVolume());

      /*--- Set distance to the surface ---*/

      numerics->SetDistance(geometry->node[iPoint]->GetWall_Distance(), 0.0);

      /*--- Menter's first blending function ---*/

      numerics->SetF1blending(nodes->GetF1blending(iPoint),0.0);

      /*--- Menter's second blending function ---*/

      numerics->SetF2blending(nodes->GetF2blending(iPoint),0.0);

      /*--- Set vorticity and strain rate magnitude ---*/

      numerics->SetStrainMag(flowNodes->GetStrainMag(iPoint), 0.0);
      numerics->SetVorticityMag(flowNodes->GetVorticityMag(iPoint), 0.0);

      /*--- Cross diffusion ---*/

      numerics->SetCrossDiff(nodes->GetCrossDiff(iPoint));
      numerics->SetCrossDiffLimited(nodes->GetCrossDiffLimited(iPoint));

      /*--- Compute the source term ---*/

      auto residual = numerics->ComputeResidual(config);

      /*--- Subtract residual and the Jacobian ---*/

      LinSysRes.SubtractBlock(iPoint, residual);
      Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);

      /*--- Compute Jacobian for gradient terms in cross-diffusion ---*/
      // if ((config->GetUse_Accurate_Turb_Jacobians()) &&
      //     (config->GetUse_Accurate_Visc_Jacobians()) && 
      //     (!nodes->GetCrossDiffLimited(iPoint)))
      //   CrossDiffusionJacobian(solver, geometry, config, iPoint);
      
    }// if dist

  }// iPoint
  
}

void CTurbSSTSolver::CrossDiffusionJacobian(CSolver         **solver,
                                            const CGeometry *geometry,
                                            const CConfig   *config,
                                            const unsigned long iPoint) {
  
  const bool wasActive = AD::BeginPassive();

  const bool gg  = config->GetKind_Gradient_Method() == GREEN_GAUSS;

  const auto node_i = geometry->node[iPoint];

  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();

  const su2double sign_grad_i = gg? 1.0 : -1.0;
  
  const su2double F1 = nodes->GetF1blending(iPoint);
  // const su2double F2 = nodes->GetF2blending(iPoint);
  // const su2double a1 = constants[7];

  const su2double r_i  = flowNodes->GetDensity(iPoint);
  const su2double om_i = nodes->GetPrimitive(iPoint,1);
  // const su2double z_i  = max(om_i, flowNodes->GetVorticityMag(iPoint)*F2/a1);

  const su2double sigma_om2 = constants[3];
  const su2double Vol       = node_i->GetVolume();

  const su2double factor = 2.0*(1. - F1)*sigma_om2*r_i/om_i*Vol;
  // const su2double factor = 2.0*(1. - F1)*sigma_om2*r_i/z_i*Vol;
  
  /*--- Reset Jacobian i and first row of Jacobian j now so we don't need to later ---*/
  Jacobian_i[0][0] = 0.; Jacobian_i[0][1] = 0.;
  Jacobian_i[1][0] = 0.; Jacobian_i[1][1] = 0.;
  Jacobian_j[0][0] = 0.; Jacobian_j[0][1] = 0.;

  /*--------------------------------------------------------------------------*/
  /*--- Step 1. Compute contributions of surface terms to the Jacobian.    ---*/
  /*---         In Green-Gauss, the weight of the surface node             ---*/
  /*---         contribution is (0.5*Sum(n_v)+n_s)/r. Least squares        ---*/
  /*---         gradients do not have a surface term.                      ---*/
  /*--------------------------------------------------------------------------*/

  su2double gradWeight[MAXNDIM] = {0.0};
  const auto gradk  = nodes->GetGradient(iPoint)[0];
  const auto gradom = nodes->GetGradient(iPoint)[1];
  if (gg && node_i->GetPhysicalBoundary()) {
    SetSurfaceGradWeights_GG(gradWeight, geometry, config, iPoint);

    Jacobian_i[1][0] = factor/r_i*GeometryToolbox::DotProduct(nDim,gradWeight,gradom);
    Jacobian_i[1][1] = factor/r_i*GeometryToolbox::DotProduct(nDim,gradWeight,gradk);
  }// if physical boundary
  
  /*--------------------------------------------------------------------------*/
  /*--- Step 2. Compute contributions of neighbor nodes to the Jacobian.   ---*/
  /*---         To prevent extra communication overhead, we only consider  ---*/
  /*---         neighbors on the current rank.                             ---*/
  /*--------------------------------------------------------------------------*/

  for (auto iNeigh = 0u; iNeigh < node_i->GetnPoint(); iNeigh++) {
    const unsigned long jPoint = node_i->GetPoint(iNeigh);
    const su2double r_j  = flowNodes->GetDensity(jPoint);
    
    SetGradWeights(gradWeight, solver[TURB_SOL], geometry, config, iPoint, jPoint);

    Jacobian_i[1][0] += factor/r_i*GeometryToolbox::DotProduct(nDim,gradWeight,gradom)*sign_grad_i;
    Jacobian_j[1][0]  = factor/r_j*GeometryToolbox::DotProduct(nDim,gradWeight,gradom);

    Jacobian_i[1][1] += factor/r_i*GeometryToolbox::DotProduct(nDim,gradWeight,gradk)*sign_grad_i;
    Jacobian_j[1][1]  = factor/r_j*GeometryToolbox::DotProduct(nDim,gradWeight,gradk);
    
    Jacobian.SubtractBlock(iPoint, jPoint, Jacobian_j);
  }// iNeigh

  Jacobian.SubtractBlock2Diag(iPoint, Jacobian_i);

  AD::EndPassive(wasActive);
  
}

void CTurbSSTSolver::Source_Template(CGeometry *geometry, CSolver **solver, CNumerics *numerics,
                                     CConfig *config, unsigned short iMesh) {
}

void CTurbSSTSolver::BC_HeatFlux_Wall(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                      CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  su2double distance, Density_Wall = 0.0;
  su2double Density_Normal = 0.0, Energy_Normal = 0.0, Kine_Normal = 0.0, Lam_Visc_Normal = 0.0;
  su2double Vel[MAXNDIM] = {0.0};
  const su2double beta_1 = constants[4];
  
  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();
  
  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    const auto node_i = geometry->node[iPoint];

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
    if (node_i->GetDomain()) {
      
      if (geometry->vertex[val_marker][iVertex]->GetDonorFound()){

        /*--- Identify the boundary by string name ---*/
        const string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

        const su2double *doubleInfo = config->GetWallFunction_DoubleInfo(Marker_Tag);
        distance = doubleInfo[0];

        /*--- Load the coefficients and interpolate ---*/
        unsigned short nDonors = geometry->vertex[val_marker][iVertex]->GetnDonorPoints();

        Density_Normal = 0.;
        Energy_Normal  = 0.;
        Kine_Normal    = 0.;
        Lam_Visc_Normal = 0.;

        for (auto iDim = 0u; iDim < nDim; iDim++) Vel[iDim] = 0.;

        for (auto iNode = 0u; iNode < nDonors; iNode++) {
          const unsigned long donorPoint = geometry->vertex[val_marker][iVertex]->GetInterpDonorPoint(iNode);
          const su2double donorCoeff     = geometry->vertex[val_marker][iVertex]->GetDonorCoeff(iNode);

          Density_Normal += donorCoeff*flowNodes->GetDensity(donorPoint);
          Energy_Normal  += donorCoeff*flowNodes->GetEnergy(donorPoint);
          Kine_Normal    += donorCoeff*nodes->GetPrimitive(donorPoint, 0);

          for (auto iDim = 0u; iDim < nDim; iDim++) Vel[iDim] += donorCoeff*flowNodes->GetVelocity(donorPoint,iDim);
        }
        const su2double VelMod = GeometryToolbox::SquaredNorm(nDim,Vel);
        Energy_Normal -= Kine_Normal + 0.5*VelMod;
        
        solver[FLOW_SOL]->GetFluidModel()->SetTDState_rhoe(Density_Normal, Energy_Normal);
        Lam_Visc_Normal = solver[FLOW_SOL]->GetFluidModel()->GetLaminarViscosity();
      }
      else {
        const unsigned long donorPoint = geometry->vertex[val_marker][iVertex]->GetNormal_Neighbor();
        
        distance = geometry->node[donorPoint]->GetWall_Distance();
        
        Density_Normal  = flowNodes->GetDensity(donorPoint);
        Lam_Visc_Normal = flowNodes->GetLaminarViscosity(donorPoint);

      }
      
      Density_Wall = flowNodes->GetDensity(iPoint);
      
      Solution[0] = 0.0;
      Solution[1] = 60.0*Density_Wall*Lam_Visc_Normal/(Density_Normal*beta_1*distance*distance);

      /*--- Set the solution values and zero the residual ---*/
      nodes->SetSolution_Old(iPoint,Solution);
      LinSysRes.SetBlock_Zero(iPoint);

      /*--- Change rows of the Jacobian (includes 1 in the diagonal) ---*/
      for (auto iVar = 0u; iVar < nVar; iVar++) {
        const unsigned long total_index = iPoint*nVar+iVar;
        Jacobian.DeleteValsRowi(total_index);
      }
    }
  }
}

void CTurbSSTSolver::BC_Isothermal_Wall(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                        CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  /*--- Call the equivalent heat flux wall boundary condition. ---*/
  BC_HeatFlux_Wall(geometry, solver, conv_numerics, visc_numerics, config, val_marker);

}

void CTurbSSTSolver::BC_Far_Field(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                  CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  const su2double *Vel_Infty = config->GetVelocity_FreeStreamND();
  const su2double Intensity = config->GetTurbulenceIntensity_FreeStream();

  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();

  for (auto iVertex = 0u; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    const auto node_i = geometry->node[iPoint];

    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/

    if (node_i->GetDomain()) {

      /*--- Allocate the value at the infinity ---*/

      const auto V_infty = solver[FLOW_SOL]->GetCharacPrimVar(val_marker, iVertex);

      /*--- Retrieve solution at the farfield boundary node ---*/

      const auto V_domain = flowNodes->GetPrimitive(iPoint);

      conv_numerics->SetPrimitive(V_domain, V_infty);

      /*--- Set turbulent variable at the wall, and at infinity ---*/

      for (auto iVar = 0u; iVar < nVar; iVar++) Primitive_i[iVar] = nodes->GetPrimitive(iPoint,iVar);

      /*--- Set Normal (it is necessary to change the sign) ---*/

      su2double Normal[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Normal[iDim] = -geometry->vertex[val_marker][iVertex]->GetNormal(iDim);
      conv_numerics->SetNormal(Normal);
      
      /*--- Set primitive state based on flow direction ---*/
      
      const su2double Vn_Infty = GeometryToolbox::DotProduct(nDim,Vel_Infty,Normal);
      if (Vn_Infty > 0.0) {
        /*--- Outflow conditions ---*/
        Primitive_j[0] = Primitive_i[0];
        Primitive_j[1] = Primitive_i[1];
      }
      else {
        /*--- Inflow conditions ---*/
        su2double Velocity2 = 0.0;
        for (auto iDim = 0u; iDim < nDim; iDim++) Velocity2 += pow(V_infty[iDim+1],2.);
        const su2double Rho_Infty = V_infty[nDim+2];
        const su2double muT_Infty = V_infty[nDim+6];
        const su2double Kine_Infty  = 1.5*Velocity2*Intensity*Intensity;
        const su2double Omega_Infty = Rho_Infty*Kine_Infty/muT_Infty;

        Primitive_j[0] = Kine_Infty;
        Primitive_j[1] = Omega_Infty;
      }
      
      conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

      /*--- Grid Movement ---*/

      if (dynamic_grid)
        conv_numerics->SetGridVel(node_i->GetGridVel(),
                                  node_i->GetGridVel());

      /*--- Compute residuals and Jacobians ---*/

      auto residual = conv_numerics->ComputeResidual(config);

      /*--- Add residuals and Jacobians ---*/

      LinSysRes.AddBlock(iPoint, residual);
      Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

      /*--- Viscous contribution ---*/
      
      visc_numerics->SetCoord(node_i->GetCoord(),
                              node_i->GetCoord());
      visc_numerics->SetNormal(Normal);

      /*--- Conservative variables w/o reconstruction ---*/
      visc_numerics->SetPrimitive(V_domain, V_domain);
      visc_numerics->SetPrimVarGradient(flowNodes->GetGradient_Primitive(iPoint),
                                        flowNodes->GetGradient_Primitive(iPoint));

      /*--- Turbulent variables w/o reconstruction, and its gradients ---*/
      visc_numerics->SetTurbVar(Primitive_i, Primitive_i);
      visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint),
                                        nodes->GetGradient(iPoint));

      /*--- Menter's first blending function ---*/
      visc_numerics->SetF1blending(nodes->GetF1blending(iPoint),
                                   nodes->GetF1blending(iPoint));
      
      /*--- Menter's second blending function ---*/
      visc_numerics->SetF2blending(nodes->GetF2blending(iPoint),
                                   nodes->GetF2blending(iPoint));
      
      /*--- Vorticity ---*/
      visc_numerics->SetVorticity(flowNodes->GetVorticity(iPoint),
                                  flowNodes->GetVorticity(iPoint));
      visc_numerics->SetStrainMag(flowNodes->GetStrainMag(iPoint),
                                  flowNodes->GetStrainMag(iPoint));
      visc_numerics->SetVorticityMag(flowNodes->GetVorticityMag(iPoint),
                                     flowNodes->GetVorticityMag(iPoint));

      /*--- Set values for gradient Jacobian ---*/
      visc_numerics->SetVolume(node_i->GetVolume(),
                               node_i->GetVolume());

      /*--- Compute residual, and Jacobians ---*/
      auto visc_residual = visc_numerics->ComputeResidual(config);

      /*--- Subtract residual, and update Jacobians ---*/
      LinSysRes.SubtractBlock(iPoint, visc_residual);

      Jacobian.SubtractBlock2Diag(iPoint, visc_residual.jacobian_i);
      if (config->GetUse_Accurate_Visc_Jacobians()) {
        CorrectViscousJacobian(solver, geometry, config, iPoint, iPoint, visc_residual.jacobianWeights_i);
      }
        
    }
  }

}

void CTurbSSTSolver::BC_Inlet(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                              CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

  /*--- Loop over all the vertices on this boundary marker ---*/

  for (auto iVertex = 0u; iVertex < geometry->nVertex[val_marker]; iVertex++) {

    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e., not a halo node) ---*/

    if (geometry->node[iPoint]->GetDomain()) {

      /*--- Normal vector for this vertex (negate for outward convention) ---*/

      su2double Normal[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Normal[iDim] = -geometry->vertex[val_marker][iVertex]->GetNormal(iDim);

      /*--- Allocate the value at the inlet ---*/

      auto V_inlet = solver[FLOW_SOL]->GetCharacPrimVar(val_marker, iVertex);

      /*--- Retrieve solution at the farfield boundary node ---*/

      auto V_domain = solver[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

      /*--- Set various quantities in the solver class ---*/

      conv_numerics->SetPrimitive(V_domain, V_inlet);

      /*--- Set the turbulent variable states. Use free-stream SST
       values for the turbulent state at the inflow. ---*/

      for (auto iVar = 0u; iVar < nVar; iVar++)
        Primitive_i[iVar] = nodes->GetPrimitive(iPoint,iVar);

      /*--- Load the inlet turbulence variables (uniform by default). ---*/

      Primitive_j[0] = Inlet_TurbVars[val_marker][iVertex][0];
      Primitive_j[1] = Inlet_TurbVars[val_marker][iVertex][1];

      conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

      /*--- Set various other quantities in the solver class ---*/

      conv_numerics->SetNormal(Normal);

      if (dynamic_grid)
        conv_numerics->SetGridVel(geometry->node[iPoint]->GetGridVel(),
                                  geometry->node[iPoint]->GetGridVel());

      /*--- Compute the residual using an upwind scheme ---*/

      auto residual = conv_numerics->ComputeResidual(config);
      LinSysRes.AddBlock(iPoint, residual);

      /*--- Jacobian contribution for implicit integration ---*/

      Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

      /*--- Viscous contribution, commented out because serious convergence problems ---*/

      visc_numerics->SetCoord(geometry->node[iPoint]->GetCoord(), geometry->node[iPoint]->GetCoord());
      visc_numerics->SetNormal(Normal);

      /*--- Conservative variables w/o reconstruction ---*/

      visc_numerics->SetPrimitive(V_domain, V_domain);

      /*--- Turbulent variables w/o reconstruction, and its gradients ---*/

      visc_numerics->SetTurbVar(Primitive_i, Primitive_i);
      visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nodes->GetGradient(iPoint));

      /*--- Menter's first blending function ---*/
      visc_numerics->SetF1blending(nodes->GetF1blending(iPoint),
                                   nodes->GetF1blending(iPoint));
      
      /*--- Menter's second blending function ---*/
      visc_numerics->SetF2blending(nodes->GetF2blending(iPoint),
                                   nodes->GetF2blending(iPoint));
      
      /*--- Vorticity ---*/
      visc_numerics->SetVorticity(solver[FLOW_SOL]->GetNodes()->GetVorticity(iPoint),
                                  solver[FLOW_SOL]->GetNodes()->GetVorticity(iPoint));

      /*--- Set values for gradient Jacobian ---*/
      visc_numerics->SetVolume(geometry->node[iPoint]->GetVolume(),
                               geometry->node[iPoint]->GetVolume());

      /*--- Compute residual, and Jacobians ---*/
      auto visc_residual = visc_numerics->ComputeResidual(config);

      /*--- Subtract residual, and update Jacobians ---*/
      LinSysRes.SubtractBlock(iPoint, visc_residual);
      Jacobian.SubtractBlock2Diag(iPoint, visc_residual.jacobian_i);

    }

  }

}

void CTurbSSTSolver::BC_Outlet(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                               CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  /*--- Loop over all the vertices on this boundary marker ---*/

  for (auto iVertex = 0u; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Check if the node belongs to the domain (i.e., not a halo node) ---*/

    if (geometry->node[iPoint]->GetDomain()) {

      /*--- Allocate the value at the outlet ---*/

      auto V_outlet = solver[FLOW_SOL]->GetCharacPrimVar(val_marker, iVertex);

      /*--- Retrieve solution at the farfield boundary node ---*/

      auto V_domain = solver[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

      /*--- Set various quantities in the solver class ---*/

      conv_numerics->SetPrimitive(V_domain, V_outlet);

      /*--- Set the turbulent variables. Here we use a Neumann BC such
       that the turbulent variable is copied from the interior of the
       domain to the outlet before computing the residual.
       Primitive_i --> TurbVar_internal,
       Primitive_j --> TurbVar_outlet ---*/

      for (auto iVar = 0u; iVar < nVar; iVar++) {
        Primitive_i[iVar] = nodes->GetPrimitive(iPoint,iVar);
        Primitive_j[iVar] = nodes->GetPrimitive(iPoint,iVar);
      }
      conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

      /*--- Set Normal (negate for outward convention) ---*/

      su2double Normal[MAXNDIM] = {0.0};
      for (auto iDim = 0u; iDim < nDim; iDim++)
        Normal[iDim] = -geometry->vertex[val_marker][iVertex]->GetNormal(iDim);
      conv_numerics->SetNormal(Normal);

      if (dynamic_grid)
      conv_numerics->SetGridVel(geometry->node[iPoint]->GetGridVel(),
                                geometry->node[iPoint]->GetGridVel());

      /*--- Compute the residual using an upwind scheme ---*/

      auto residual = conv_numerics->ComputeResidual(config);
      LinSysRes.AddBlock(iPoint, residual);

      /*--- Jacobian contribution for implicit integration ---*/

      Jacobian.AddBlock2Diag(iPoint, residual.jacobian_i);

//      /*--- Viscous contribution, commented out because serious convergence problems ---*/
//
//      visc_numerics->SetCoord(geometry->node[iPoint]->GetCoord(), geometry->node[iPoint]->GetCoord());
//      visc_numerics->SetNormal(Normal);
//
//      /*--- Conservative variables w/o reconstruction ---*/
//
//      visc_numerics->SetPrimitive(V_domain, V_domain);
//
//      /*--- Turbulent variables w/o reconstruction, and its gradients ---*/
//
//      visc_numerics->SetTurbVar(Primitive_i, Primitive_i);
//      visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nodes->GetGradient(iPoint));
//
//      /*--- Menter's first blending function ---*/
//      visc_numerics->SetF1blending(nodes->GetF1blending(iPoint),
//                                   nodes->GetF1blending(iPoint));
//
//      /*--- Menter's second blending function ---*/
//      visc_numerics->SetF2blending(nodes->GetF2blending(iPoint),
//                                   nodes->GetF2blending(iPoint));
//
//      /*--- Vorticity ---*/
//      visc_numerics->SetVorticity(solver[FLOW_SOL]->GetNodes()->GetVorticity(iPoint),
//                                  solver[FLOW_SOL]->GetNodes()->GetVorticity(iPoint));
//
//      /*--- Set values for gradient Jacobian ---*/
//      visc_numerics->SetVolume(geometry->node[iPoint]->GetVolume(),
//                               geometry->node[iPoint]->GetVolume());
//
//      /*--- Compute residual, and Jacobians ---*/
//      auto visc_residual = visc_numerics->ComputeResidual(config);
//
//      /*--- Subtract residual, and update Jacobians ---*/
//      LinSysRes.SubtractBlock(iPoint, visc_residual);
//      Jacobian.SubtractBlock2Diag(iPoint, visc_residual.jacobian_i);

    }
  }

}


void CTurbSSTSolver::BC_Inlet_MixingPlane(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                          CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  unsigned short iVar, iSpan, iDim;
  unsigned long  oldVertex, iPoint, Point_Normal, iVertex;
  su2double *V_inlet, *V_domain, *Normal;
  su2double extAverageKine, extAverageOmega;
  unsigned short nSpanWiseSections = config->GetnSpanWiseSections();

  Normal = new su2double[nDim];

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);

  /*--- Loop over all the vertices on this boundary marker ---*/
  for (iSpan= 0; iSpan < nSpanWiseSections ; iSpan++){
    extAverageKine = solver[FLOW_SOL]->GetExtAverageKine(val_marker, iSpan);
    extAverageOmega = solver[FLOW_SOL]->GetExtAverageOmega(val_marker, iSpan);


    /*--- Loop over all the vertices on this boundary marker ---*/

    for (iVertex = 0; iVertex < geometry->GetnVertexSpan(val_marker,iSpan); iVertex++) {

      /*--- find the node related to the vertex ---*/
      iPoint = geometry->turbovertex[val_marker][iSpan][iVertex]->GetNode();

      /*--- using the other vertex information for retrieving some information ---*/
      oldVertex = geometry->turbovertex[val_marker][iSpan][iVertex]->GetOldVertex();

      /*--- Index of the closest interior node ---*/
      Point_Normal = geometry->vertex[val_marker][oldVertex]->GetNormal_Neighbor();

      /*--- Normal vector for this vertex (negate for outward convention) ---*/

      geometry->vertex[val_marker][oldVertex]->GetNormal(Normal);
      for (iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];

      /*--- Allocate the value at the inlet ---*/
      V_inlet = solver[FLOW_SOL]->GetCharacPrimVar(val_marker, oldVertex);

      /*--- Retrieve solution at the farfield boundary node ---*/

      V_domain = solver[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

      /*--- Set various quantities in the solver class ---*/

      conv_numerics->SetPrimitive(V_domain, V_inlet);

      /*--- Set the turbulent variable states (prescribed for an inflow) ---*/

      for (iVar = 0; iVar < nVar; iVar++)
        Primitive_i[iVar] = nodes->GetPrimitive(iPoint,iVar);

      Primitive_j[0]= extAverageKine;
      Primitive_j[1]= extAverageOmega;

      conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

      /*--- Set various other quantities in the solver class ---*/
      conv_numerics->SetNormal(Normal);

      if (dynamic_grid)
        conv_numerics->SetGridVel(geometry->node[iPoint]->GetGridVel(),
            geometry->node[iPoint]->GetGridVel());

      /*--- Compute the residual using an upwind scheme ---*/
      auto conv_residual = conv_numerics->ComputeResidual(config);

      /*--- Jacobian contribution for implicit integration ---*/
      LinSysRes.AddBlock(iPoint, conv_residual);
      Jacobian.AddBlock2Diag(iPoint, conv_residual.jacobian_i);

      /*--- Viscous contribution ---*/
      visc_numerics->SetCoord(geometry->node[iPoint]->GetCoord(), geometry->node[Point_Normal]->GetCoord());
      visc_numerics->SetNormal(Normal);

      /*--- Conservative variables w/o reconstruction ---*/
      visc_numerics->SetPrimitive(V_domain, V_inlet);

      /*--- Turbulent variables w/o reconstruction, and its gradients ---*/
      visc_numerics->SetTurbVar(Primitive_i, Primitive_j);
      visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nodes->GetGradient(iPoint));

      /*--- Menter's first blending function ---*/
      visc_numerics->SetF1blending(nodes->GetF1blending(iPoint), nodes->GetF1blending(iPoint));

      /*--- Compute residual, and Jacobians ---*/
      auto visc_residual = visc_numerics->ComputeResidual(config);

      /*--- Subtract residual, and update Jacobians ---*/
      LinSysRes.SubtractBlock(iPoint, visc_residual);
      Jacobian.SubtractBlock2Diag(iPoint, visc_residual.jacobian_i);

    }
  }

  /*--- Free locally allocated memory ---*/
  delete[] Normal;

}

void CTurbSSTSolver::BC_Inlet_Turbo(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                    CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {

  unsigned short iVar, iSpan, iDim;
  unsigned long  oldVertex, iPoint, Point_Normal, iVertex;
  su2double *V_inlet, *V_domain, *Normal;
  unsigned short nSpanWiseSections = config->GetnSpanWiseSections();

  /*--- Quantities for computing the  kine and omega to impose at the inlet boundary. ---*/
  su2double rho, pressure, *Vel, VelMag, muLam, Intensity, viscRatio, kine_b, omega_b, kine;
  CFluidModel *FluidModel;

  FluidModel = solver[FLOW_SOL]->GetFluidModel();
  Intensity = config->GetTurbulenceIntensity_FreeStream();
  viscRatio = config->GetTurb2LamViscRatio_FreeStream();

  Normal = new su2double[nDim];
  Vel = new su2double[nDim];

  string Marker_Tag = config->GetMarker_All_TagBound(val_marker);


  for (iSpan= 0; iSpan < nSpanWiseSections ; iSpan++){

    /*--- Compute the inflow kine and omega using the span wise averge quntities---*/
    for (iDim = 0; iDim < nDim; iDim++)
      Vel[iDim] = solver[FLOW_SOL]->GetAverageTurboVelocity(val_marker, iSpan)[iDim];

    rho       = solver[FLOW_SOL]->GetAverageDensity(val_marker, iSpan);
    pressure  = solver[FLOW_SOL]->GetAveragePressure(val_marker, iSpan);
    kine      = solver[FLOW_SOL]->GetAverageKine(val_marker, iSpan);

    FluidModel->SetTDState_Prho(pressure, rho);
    muLam = FluidModel->GetLaminarViscosity();

    VelMag = 0;
    for (iDim = 0; iDim < nDim; iDim++)
      VelMag += Vel[iDim]*Vel[iDim];
    VelMag = sqrt(VelMag);

    kine_b  = 3.0/2.0*(VelMag*VelMag*Intensity*Intensity);
    omega_b = rho*kine/(muLam*viscRatio);

    /*--- Loop over all the vertices on this boundary marker ---*/
    for (iVertex = 0; iVertex < geometry->GetnVertexSpan(val_marker,iSpan); iVertex++) {

      /*--- find the node related to the vertex ---*/
      iPoint = geometry->turbovertex[val_marker][iSpan][iVertex]->GetNode();

      /*--- using the other vertex information for retrieving some information ---*/
      oldVertex = geometry->turbovertex[val_marker][iSpan][iVertex]->GetOldVertex();

      /*--- Index of the closest interior node ---*/
      Point_Normal = geometry->vertex[val_marker][oldVertex]->GetNormal_Neighbor();

      /*--- Normal vector for this vertex (negate for outward convention) ---*/

      geometry->vertex[val_marker][oldVertex]->GetNormal(Normal);
      for (iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];

      /*--- Allocate the value at the inlet ---*/
      V_inlet = solver[FLOW_SOL]->GetCharacPrimVar(val_marker, oldVertex);

      /*--- Retrieve solution at the farfield boundary node ---*/

      V_domain = solver[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint);

      /*--- Set various quantities in the solver class ---*/

      conv_numerics->SetPrimitive(V_domain, V_inlet);

      for (iVar = 0; iVar < nVar; iVar++)
        Primitive_i[iVar] = nodes->GetPrimitive(iPoint,iVar);

      /*--- Set the turbulent variable states. Use average span-wise values
             values for the turbulent state at the inflow. ---*/

      Primitive_j[0]= kine_b;
      Primitive_j[1]= omega_b;

      conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

      /*--- Set various other quantities in the solver class ---*/
      conv_numerics->SetNormal(Normal);

      if (dynamic_grid)
        conv_numerics->SetGridVel(geometry->node[iPoint]->GetGridVel(),
                                  geometry->node[iPoint]->GetGridVel());

      /*--- Compute the residual using an upwind scheme ---*/
      auto conv_residual = conv_numerics->ComputeResidual(config);

      /*--- Jacobian contribution for implicit integration ---*/
      LinSysRes.AddBlock(iPoint, conv_residual);
      Jacobian.AddBlock2Diag(iPoint, conv_residual.jacobian_i);

      /*--- Viscous contribution ---*/
      visc_numerics->SetCoord(geometry->node[iPoint]->GetCoord(), geometry->node[Point_Normal]->GetCoord());
      visc_numerics->SetNormal(Normal);

      /*--- Conservative variables w/o reconstruction ---*/
      visc_numerics->SetPrimitive(V_domain, V_inlet);

      /*--- Turbulent variables w/o reconstruction, and its gradients ---*/
      visc_numerics->SetTurbVar(Primitive_i, Primitive_j);
      visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nodes->GetGradient(iPoint));

      /*--- Menter's first blending function ---*/
      visc_numerics->SetF1blending(nodes->GetF1blending(iPoint), nodes->GetF1blending(iPoint));

      /*--- Compute residual, and Jacobians ---*/
      auto visc_residual = visc_numerics->ComputeResidual(config);

      /*--- Subtract residual, and update Jacobians ---*/
      LinSysRes.SubtractBlock(iPoint, visc_residual);
      Jacobian.SubtractBlock2Diag(iPoint, visc_residual.jacobian_i);

    }
  }

  /*--- Free locally allocated memory ---*/
  delete[] Normal;
  delete[] Vel;

}

void CTurbSSTSolver::BC_Fluid_Interface(CGeometry *geometry, CSolver **solver, CNumerics *conv_numerics,
                                        CNumerics *visc_numerics, CConfig *config){

  unsigned long iVertex, jVertex, iPoint, Point_Normal = 0;
  unsigned short iDim, iVar, jVar, iMarker;

  unsigned short nPrimVar = solver[FLOW_SOL]->GetnPrimVar();
  su2double *Normal = new su2double[nDim];
  su2double *PrimVar_i = new su2double[nPrimVar];
  su2double *PrimVar_j = new su2double[nPrimVar];

  unsigned long nDonorVertex;
  su2double weight;

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    if (config->GetMarker_All_KindBC(iMarker) == FLUID_INTERFACE) {

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        Point_Normal = geometry->vertex[iMarker][iVertex]->GetNormal_Neighbor();

        if (geometry->node[iPoint]->GetDomain()) {

          nDonorVertex = GetnSlidingStates(iMarker, iVertex);

          /*--- Initialize Residual, this will serve to accumulate the average ---*/

          for (iVar = 0; iVar < nVar; iVar++) {
            Residual[iVar] = 0.0;
            for (jVar = 0; jVar < nVar; jVar++)
              Jacobian_i[iVar][jVar] = 0.0;
          }

          /*--- Loop over the nDonorVertexes and compute the averaged flux ---*/

          for (jVertex = 0; jVertex < nDonorVertex; jVertex++){

            geometry->vertex[iMarker][iVertex]->GetNormal(Normal);
            for (iDim = 0; iDim < nDim; iDim++) Normal[iDim] = -Normal[iDim];

            for (iVar = 0; iVar < nPrimVar; iVar++) {
              PrimVar_i[iVar] = solver[FLOW_SOL]->GetNodes()->GetPrimitive(iPoint,iVar);
              PrimVar_j[iVar] = solver[FLOW_SOL]->GetSlidingState(iMarker, iVertex, iVar, jVertex);
            }

            /*--- Get the weight computed in the interpolator class for the j-th donor vertex ---*/

            weight = solver[FLOW_SOL]->GetSlidingState(iMarker, iVertex, nPrimVar, jVertex);

            /*--- Set primitive variables ---*/

            conv_numerics->SetPrimitive( PrimVar_i, PrimVar_j );

            /*--- Set the turbulent variable states ---*/
            Primitive_i[0] = nodes->GetPrimitive(iPoint,0);
            Primitive_i[1] = nodes->GetPrimitive(iPoint,1);

            Primitive_j[0] = GetSlidingState(iMarker, iVertex, 0, jVertex);
            Primitive_j[1] = GetSlidingState(iMarker, iVertex, 1, jVertex);

            conv_numerics->SetTurbVar(Primitive_i, Primitive_j);

            /*--- Set the normal vector ---*/

            conv_numerics->SetNormal(Normal);

            if (dynamic_grid)
              conv_numerics->SetGridVel(geometry->node[iPoint]->GetGridVel(), geometry->node[iPoint]->GetGridVel());

            auto residual = conv_numerics->ComputeResidual(config);

            /*--- Accumulate the residuals to compute the average ---*/

            for (iVar = 0; iVar < nVar; iVar++) {
              Residual[iVar] += weight*residual.residual[iVar];
              for (jVar = 0; jVar < nVar; jVar++)
                Jacobian_i[iVar][jVar] += weight*residual.jacobian_i[iVar][jVar];
            }
          }

          /*--- Add Residuals and Jacobians ---*/

          LinSysRes.AddBlock(iPoint, Residual);

          Jacobian.AddBlock2Diag(iPoint, Jacobian_i);

          /*--- Set the normal vector and the coordinates ---*/

          visc_numerics->SetNormal(Normal);
          visc_numerics->SetCoord(geometry->node[iPoint]->GetCoord(), geometry->node[Point_Normal]->GetCoord());

          /*--- Primitive variables, and gradient ---*/

          visc_numerics->SetPrimitive(PrimVar_i, PrimVar_j);
          //          visc_numerics->SetPrimVarGradient(node[iPoint]->GetGradient_Primitive(), node[iPoint]->GetGradient_Primitive());

          /*--- Turbulent variables and its gradients  ---*/

          visc_numerics->SetTurbVar(Primitive_i, Primitive_j);
          visc_numerics->SetTurbVarGradient(nodes->GetGradient(iPoint), nodes->GetGradient(iPoint));

          /*--- Compute and update residual ---*/

          auto residual = visc_numerics->ComputeResidual(config);

          LinSysRes.SubtractBlock(iPoint, residual);

          /*--- Jacobian contribution for implicit integration ---*/

          Jacobian.SubtractBlock2Diag(iPoint, residual.jacobian_i);

        }
      }
    }
  }

  /*--- Free locally allocated memory ---*/

  delete [] Normal;
  delete [] PrimVar_i;
  delete [] PrimVar_j;

}

void CTurbSSTSolver::SetInletAtVertex(su2double *val_inlet,
                                     unsigned short iMarker,
                                     unsigned long iVertex) {

  Inlet_TurbVars[iMarker][iVertex][0] = val_inlet[nDim+2+nDim];
  Inlet_TurbVars[iMarker][iVertex][1] = val_inlet[nDim+2+nDim+1];

}

su2double CTurbSSTSolver::GetInletAtVertex(su2double *val_inlet,
                                           unsigned long val_inlet_point,
                                           unsigned short val_kind_marker,
                                           string val_marker,
                                           CGeometry *geometry,
                                           CConfig *config) const {

  /*--- Local variables ---*/

  unsigned short iMarker, iDim;
  unsigned long iPoint, iVertex;
  su2double Area = 0.0;
  su2double Normal[3] = {0.0,0.0,0.0};

  /*--- Alias positions within inlet file for readability ---*/

  if (val_kind_marker == INLET_FLOW) {

    unsigned short tke_position   = nDim+2+nDim;
    unsigned short omega_position = nDim+2+nDim+1;

    for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
      if ((config->GetMarker_All_KindBC(iMarker) == INLET_FLOW) &&
          (config->GetMarker_All_TagBound(iMarker) == val_marker)) {

        for (iVertex = 0; iVertex < nVertex[iMarker]; iVertex++){

          iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

          if (iPoint == val_inlet_point) {

            /*-- Compute boundary face area for this vertex. ---*/

            geometry->vertex[iMarker][iVertex]->GetNormal(Normal);
            Area = 0.0;
            for (iDim = 0; iDim < nDim; iDim++)
              Area += Normal[iDim]*Normal[iDim];
            Area = sqrt(Area);

            /*--- Access and store the inlet variables for this vertex. ---*/

            val_inlet[tke_position]   = Inlet_TurbVars[iMarker][iVertex][0];
            val_inlet[omega_position] = Inlet_TurbVars[iMarker][iVertex][1];

            /*--- Exit once we find the point. ---*/

            return Area;

          }
        }
      }
    }

  }

  /*--- If we don't find a match, then the child point is not on the
   current inlet boundary marker. Return zero area so this point does
   not contribute to the restriction operator and continue. ---*/

  return Area;

}

void CTurbSSTSolver::SetUniformInlet(CConfig* config, unsigned short iMarker) {

  for(unsigned long iVertex=0; iVertex < nVertex[iMarker]; iVertex++){
    Inlet_TurbVars[iMarker][iVertex][0] = kine_Inf;
    Inlet_TurbVars[iMarker][iVertex][1] = omega_Inf;
  }

}

void CTurbSSTSolver::SetTime_Step(CGeometry *geometry, CSolver **solver, CConfig *config,
                                unsigned short iMesh, unsigned long Iteration) {

  const su2double K_v = 0.25;
  const su2double sigma_k1  = constants[0];
  const su2double sigma_k2  = constants[1];
  const su2double sigma_om1 = constants[2];
  const su2double sigma_om2 = constants[3];

  /*--- Init thread-shared variables to compute min/max values.
   *    Critical sections are used for this instead of reduction
   *    clauses for compatibility with OpenMP 2.0 (Windows...). ---*/

  SU2_OMP_MASTER
  {
    Min_Delta_Time = 1e30;
    Max_Delta_Time = 0.0;
  }
  SU2_OMP_BARRIER

  su2double Mean_SoundSpeed, Mean_ProjVel, Lambda, Local_Delta_Time;

  CVariable *flowNodes = solver[FLOW_SOL]->GetNodes();

  /*--- Loop domain points. ---*/

  SU2_OMP_FOR_DYN(omp_chunk_size)
  for (auto iPoint = 0ul; iPoint < nPointDomain; ++iPoint) {

    auto node_i = geometry->node[iPoint];

    /*--- Set maximum eigenvalues to zero. ---*/
    nodes->SetMax_Lambda_Inv(iPoint,0.0);
    nodes->SetMax_Lambda_Visc(iPoint,0.0);

    const auto Vol = geometry->node[iPoint]->GetVolume();
    if (Vol == 0.0) continue;

    /*--- Loop over the neighbors of point i. ---*/

    for (auto iNeigh = 0u; iNeigh < node_i->GetnPoint(); ++iNeigh)
    {
      const auto jPoint = node_i->GetPoint(iNeigh);
      const auto node_j = geometry->node[jPoint];

      const auto iEdge = node_i->GetEdge(iNeigh);
      const auto Normal = geometry->edge[iEdge]->GetNormal();
      const su2double Area = GeometryToolbox::Norm(nDim, Normal);

      /*--- Mean Values ---*/

      Mean_ProjVel = 0.5 * (flowNodes->GetProjVel(iPoint,Normal) + flowNodes->GetProjVel(jPoint,Normal));
      Mean_SoundSpeed = 0.5 * (flowNodes->GetSoundSpeed(iPoint) + flowNodes->GetSoundSpeed(jPoint)) * Area;

      /*--- Adjustment for grid movement ---*/

      if (dynamic_grid) {
        const su2double *GridVel_i = node_i->GetGridVel();
        const su2double *GridVel_j = node_j->GetGridVel();

        for (auto iDim = 0u; iDim < nDim; iDim++)
          Mean_ProjVel -= 0.5 * (GridVel_i[iDim] + GridVel_j[iDim]) * Normal[iDim];
      }

      /*--- Inviscid contribution ---*/

      Lambda = fabs(Mean_ProjVel) + Mean_SoundSpeed ;
      nodes->AddMax_Lambda_Inv(iPoint,Lambda);

      /*--- Viscous contribution ---*/

      const su2double F1_i = nodes->GetF1blending(iPoint);
      const su2double F1_j = nodes->GetF1blending(jPoint);

      const su2double sigma_k_i  = F1_i*sigma_k1  + (1.0 - F1_i)*sigma_k2;
      const su2double sigma_k_j  = F1_j*sigma_k1  + (1.0 - F1_j)*sigma_k2;
      const su2double sigma_om_i = F1_i*sigma_om1 + (1.0 - F1_i)*sigma_om2;
      const su2double sigma_om_j = F1_j*sigma_om1 + (1.0 - F1_j)*sigma_om2;

      const su2double visc_k_i  = flowNodes->GetLaminarViscosity(iPoint) + nodes->GetmuT(iPoint)*sigma_k_i;
      const su2double visc_k_j  = flowNodes->GetLaminarViscosity(jPoint) + nodes->GetmuT(jPoint)*sigma_k_j;
      const su2double visc_om_i = flowNodes->GetLaminarViscosity(iPoint) + nodes->GetmuT(iPoint)*sigma_om_i;
      const su2double visc_om_j = flowNodes->GetLaminarViscosity(jPoint) + nodes->GetmuT(jPoint)*sigma_om_j;

      const su2double Mean_Visc    = 0.5*(visc_k_i+visc_k_j + visc_om_i+visc_om_j);
      const su2double Mean_Density = 0.5*(flowNodes->GetDensity(iPoint) + flowNodes->GetDensity(jPoint));

      Lambda = Mean_Visc*Area*Area/(K_v*Mean_Density*Vol);
      nodes->AddMax_Lambda_Visc(iPoint, Lambda);
    }
  }

  /*--- Loop boundary edges ---*/

  for (auto iMarker = 0; iMarker < geometry->GetnMarker(); iMarker++) {
    if ((config->GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {

      SU2_OMP_FOR_STAT(OMP_MIN_SIZE)
      for (auto iVertex = 0ul; iVertex < geometry->GetnVertex(iMarker); iVertex++) {

        /*--- Point identification, Normal vector and area ---*/

        const auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        const auto node_i = geometry->node[iPoint];

        if (!node_i->GetDomain()) continue;

        const auto Vol = node_i->GetVolume();
        if (Vol == 0.0) continue;

        const auto Normal = geometry->vertex[iMarker][iVertex]->GetNormal();
        const su2double Area = GeometryToolbox::Norm(nDim, Normal);

        /*--- Mean Values ---*/

        Mean_ProjVel    = flowNodes->GetProjVel(iPoint,Normal);
        Mean_SoundSpeed = flowNodes->GetSoundSpeed(iPoint) * Area;

        /*--- Adjustment for grid movement ---*/

        if (dynamic_grid) {
          const su2double *GridVel = node_i->GetGridVel();

          for (auto iDim = 0u; iDim < nDim; iDim++)
            Mean_ProjVel -= GridVel[iDim]*Normal[iDim];
        }

        /*--- Inviscid contribution ---*/

        Lambda = fabs(Mean_ProjVel) + Mean_SoundSpeed;
        nodes->AddMax_Lambda_Inv(iPoint,Lambda);

        /*--- Viscous contribution ---*/

        const su2double F1_i = nodes->GetF1blending(iPoint);

        const su2double sigma_k_i  = F1_i*sigma_k1  + (1.0 - F1_i)*sigma_k2;
        const su2double sigma_om_i = F1_i*sigma_om1 + (1.0 - F1_i)*sigma_om2;

        const su2double visc_k_i  = flowNodes->GetLaminarViscosity(iPoint) + nodes->GetmuT(iPoint)*sigma_k_i;
        const su2double visc_om_i = flowNodes->GetLaminarViscosity(iPoint) + nodes->GetmuT(iPoint)*sigma_om_i;

        const su2double Mean_Visc    = (visc_k_i + visc_om_i);
        const su2double Mean_Density = flowNodes->GetDensity(iPoint);

        Lambda = Mean_Visc*Area*Area/(K_v*Mean_Density*Vol);
        nodes->AddMax_Lambda_Visc(iPoint, Lambda);

      }
    }
  }

  /*--- Each element uses their own speed, steady state simulation. ---*/
  {
    /*--- Thread-local variables for min/max reduction. ---*/
    su2double minDt = 1e30, maxDt = 0.0;

    SU2_OMP(for schedule(static,omp_chunk_size) nowait)
    for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {

      const auto Vol = geometry->node[iPoint]->GetVolume();

      if (Vol != 0.0) {

        const su2double Lambda_Inv  = nodes->GetMax_Lambda_Inv(iPoint);
        const su2double Lambda_Visc = nodes->GetMax_Lambda_Visc(iPoint);
        Lambda = (Lambda_Inv + Lambda_Visc);
        Local_Delta_Time = nodes->GetLocalCFL(iPoint)*Vol/Lambda;

        minDt = min(minDt, Local_Delta_Time);
        maxDt = max(maxDt, Local_Delta_Time);

        nodes->SetDelta_Time(iPoint, min(Local_Delta_Time, config->GetMax_DeltaTime()));
      }
      else {
        nodes->SetDelta_Time(iPoint,0.0);
      }
    }
    /*--- Min/max over threads. ---*/
    SU2_OMP_CRITICAL
    {
      Min_Delta_Time = min(Min_Delta_Time, minDt);
      Max_Delta_Time = max(Max_Delta_Time, maxDt);
    }
    SU2_OMP_BARRIER
  }

  /*--- Compute the min/max dt (in parallel, now over mpi ranks). ---*/

  SU2_OMP_MASTER
  if (config->GetComm_Level() == COMM_FULL) {
    su2double rbuf_time;
    SU2_MPI::Allreduce(&Min_Delta_Time, &rbuf_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    Min_Delta_Time = rbuf_time;

    SU2_MPI::Allreduce(&Max_Delta_Time, &rbuf_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    Max_Delta_Time = rbuf_time;
  }
  SU2_OMP_BARRIER

  /*--- TODO: For exact time solution use the minimum delta time of the whole mesh. ---*/

}

void CTurbSSTSolver::ComputeNicholsWallFunction(CGeometry *geometry, CSolver **solver, CConfig *config) {
  
  su2double Lam_Visc_Normal = 0.0, Eddy_Visc = 0.0, VelMod = 0.0;
  
  const su2double kappa = 0.41;
  const su2double B = 5.0;
  const su2double Gas_Constant = config->GetGas_ConstantND();
  const su2double Cp = (Gamma / Gamma_Minus_One) * Gas_Constant;
  const su2double Recovery = pow(config->GetPrandtl_Lam(), (1.0/3.0));
  
  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();
  
  /*--- Set K and Omega at the first point of the wall ---*/
  for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {

    const auto node_i = geometry->node[iPoint];

    if (node_i->GetBool_Wall_Neighbor() && flowNodes->GetUseWallFunction(iPoint)) {
      
      /*--- Properties at the wall from CNSSolver::ComputeWallFunction() ---*/
      su2double Density_Wall  = 0.;
      su2double Lam_Visc_Wall = 0.;
      su2double U_Tau = 0.;
      su2double T_Wall = 0.;
      const unsigned short nDonors = node_i->GetWall_nNode();
      for (auto iNode = 0u; iNode < nDonors; iNode++) {
        const su2double donorCoeff = node_i->GetWall_Interpolation_Weights()[iNode];
        Density_Wall  += donorCoeff*flowNodes->GetWallDensity(iPoint, iNode);
        Lam_Visc_Wall += donorCoeff*flowNodes->GetWallLamVisc(iPoint, iNode);
        U_Tau         += donorCoeff*flowNodes->GetWallUTau(iPoint, iNode);
        T_Wall        += donorCoeff*flowNodes->GetWallTemp(iPoint, iNode);
      }
      
      /*--- Wall function ---*/
      const su2double Gam = Recovery*U_Tau*U_Tau/(2.0*Cp*T_Wall);
      const su2double Beta = 0.0; // For adiabatic flows only
      const su2double Q    = sqrt(Beta*Beta + 4.0*Gam);
      const su2double Phi  = asin(-1.0*Beta/Q);

      VelMod = 0.;
      for (auto iDim = 0u; iDim < nDim; iDim++) VelMod += pow(flowNodes->GetVelocity(iPoint,iDim), 2.);
      VelMod = sqrt(VelMod);

      const su2double Up  = VelMod/U_Tau;
      const su2double Ypw = exp((kappa/sqrt(Gam))*(asin((2.0*Gam*Up - Beta)/Q) - Phi))*exp(-1.0*kappa*B);
      
      const su2double Yp = Up + Ypw - (exp(-1.0*kappa*B)*
                           (1.0 + kappa*Up + kappa*kappa*Up*Up/2.0 +
                           kappa*kappa*kappa*Up*Up*Up/6.0));
      
      /*--- Disable calculation if Y+ is too small or large ---*/
      if (Yp < 5.0 || Yp > 1.0e3) continue;
      
      const su2double dYpw_dYp = 2.0*Ypw*(kappa*sqrt(Gam)/Q)*pow(1.0 - pow(2.0*Gam*Up - Beta,2.0)/(Q*Q), -0.5);

      Lam_Visc_Normal = flowNodes->GetLaminarViscosity(iPoint);
      Eddy_Visc = Lam_Visc_Wall*(1.0 + dYpw_dYp - kappa*exp(-1.0*kappa*B)*
                  (1.0 + kappa*Up + kappa*kappa*Up*Up/2.0)
                  - Lam_Visc_Normal/Lam_Visc_Wall);
      
//      Eddy_Visc = flowNodes->GetEddyViscosity(iPoint);
      
      const su2double Density_Normal = flowNodes->GetDensity(iPoint);
      const su2double distance = node_i->GetWall_Distance();
      const su2double Omega_i = 6. * Lam_Visc_Wall / (0.075 * Density_Wall * pow(distance, 2.0));
      const su2double Omega_0 = U_Tau / (0.3 * 0.41 * distance);
      const su2double Omega = sqrt(pow(Omega_0, 2.) + pow(Omega_i, 2.));
      // const su2double Omega_b1 = Omega_i + Omega_0;
      // const su2double Omega_b2 = pow(pow(Omega_i, 1.2) + pow(Omega_0, 1.2), 1./1.2);
      // const su2double blend = tanh(pow(Yp/10., 4.));
      // const su2double Omega = blend*Omega_b1 + (1.-blend)*Omega_b2;

      Solution[0] = Omega * Eddy_Visc;
      Solution[1] = Density_Normal * Omega;
      
      nodes->SetSolution_Old(iPoint,Solution);
      nodes->SetSolution(iPoint,Solution);
      // LinSysRes.SetBlock_Zero(iPoint);

      // /*--- Change rows of the Jacobian (includes 1 in the diagonal) ---*/
      // for (auto iVar = 0u; iVar < nVar; iVar++) {
      //   const unsigned long total_index = iPoint*nVar+iVar;
      //   Jacobian.DeleteValsRowi(total_index);
      // }
      
    }
  }
}

void CTurbSSTSolver::ComputeKnoppWallFunction(CGeometry *geometry, CSolver **solver, CConfig *config) {
  
  su2double Eddy_Visc = 0.0, VelMod = 0.0;
  
  const su2double kappa = 0.41;
  const su2double B = 5.0;
  
  CVariable* flowNodes = solver[FLOW_SOL]->GetNodes();
  
  /*--- Set K and Omega at the first point of the wall ---*/
  for (auto iPoint = 0ul; iPoint < nPointDomain; iPoint++) {

    const auto node_i = geometry->node[iPoint];

    if (node_i->GetBool_Wall_Neighbor() && flowNodes->GetUseWallFunction(iPoint)) {
      
      /*--- Properties at the wall from CNSSolver::ComputeWallFunction() ---*/
      su2double Density_Wall  = 0.;
      su2double Lam_Visc_Wall = 0.;
      su2double U_Tau = 0.;
      const unsigned short nDonors = node_i->GetWall_nNode();
      for (auto iNode = 0u; iNode < nDonors; iNode++) {
        const su2double donorCoeff = node_i->GetWall_Interpolation_Weights()[iNode];
        Density_Wall  += donorCoeff*flowNodes->GetWallDensity(iPoint, iNode);
        Lam_Visc_Wall += donorCoeff*flowNodes->GetWallLamVisc(iPoint, iNode);
        U_Tau         += donorCoeff*flowNodes->GetWallUTau(iPoint, iNode);
      }

      VelMod = 0.;
      for (auto iDim = 0u; iDim < nDim; iDim++) VelMod += pow(flowNodes->GetVelocity(iPoint,iDim), 2.);
      VelMod = sqrt(VelMod);

      const su2double distance = node_i->GetWall_Distance();

      const su2double Up = VelMod/U_Tau;
      const su2double Yp = Density_Wall * U_Tau * distance / Lam_Visc_Wall;
      
      /*--- Disable calculation if Y+ is too small or large ---*/
      if (Yp > 500.) continue;
      
      const su2double Density_Normal = flowNodes->GetDensity(iPoint);
      const su2double Omega_i = 6. * Lam_Visc_Wall / (0.075 * Density_Wall * pow(distance, 2.0));
      const su2double Omega_0 = U_Tau / (0.3 * 0.41 * distance);
      const su2double Omega_b1 = Omega_i + Omega_0;
      const su2double Omega_b2 = pow(pow(Omega_i, 1.2) + pow(Omega_0, 1.2), 1./1.2);
      const su2double blend = tanh(pow(Yp/10., 4.));
      const su2double Omega = blend*Omega_b1 + (1.-blend)*Omega_b2;


      const su2double Eddy_Lam_Ratio = kappa * exp(-kappa * B) * 
                                       (exp(-kappa * Up) - 1. - kappa * Up - pow(kappa * Up, 2.) / 2.);

      
      Eddy_Visc = Lam_Visc_Wall * Eddy_Lam_Ratio;

      Solution[0] = Omega * Eddy_Visc;
      Solution[1] = Density_Normal * Omega;
      
      nodes->SetSolution_Old(iPoint,Solution);
      nodes->SetSolution(iPoint,Solution);
      
    }
  }

  /*--- MPI solution ---*/
  InitiateComms(geometry, config, SOLUTION);
  CompleteComms(geometry, config, SOLUTION);

  /*--- Recompute primitives, gradients, blending functions, viscosity ---*/
  Postprocessing(geometry, solver, config, MESH_0);
}

void CTurbSSTSolver::TurbulentMetric(CSolver                    **solver,
                                     CGeometry                  *geometry,
                                     CConfig                    *config,
                                     unsigned long              iPoint,
                                     vector<vector<su2double> > &weights) {

  CVariable *varFlo    = solver[FLOW_SOL]->GetNodes(),
            *varTur    = solver[TURB_SOL]->GetNodes(),
            *varAdjFlo = solver[ADJFLOW_SOL]->GetNodes(),
            *varAdjTur = solver[ADJTURB_SOL]->GetNodes();

  unsigned short iDim, jDim, iVar;
  const unsigned short nVarFlo = solver[FLOW_SOL]->GetnVar();
  const unsigned short nVarTur = solver[TURB_SOL]->GetnVar();
  
  //--- First-order terms (error due to viscosity)
  su2double r, u[3], k, omega,
            mu, mut,
            R, cp, g, Prt,
            walldist;

  r = varFlo->GetDensity(iPoint);
  u[0] = varFlo->GetVelocity(iPoint, 0);
  u[1] = varFlo->GetVelocity(iPoint, 1);
  if (nDim == 3) u[2] = varFlo->GetVelocity(iPoint, 2);
  k = varTur->GetPrimitive(iPoint, 0);
  omega = varTur->GetPrimitive(iPoint, 1);
  
  mu  = varFlo->GetLaminarViscosity(iPoint);
  mut = nodes->GetmuT(iPoint);

  g    = config->GetGamma();
  R    = config->GetGas_ConstantND();
  cp   = (g/(g-1.))*R;
  Prt  = config->GetPrandtl_Turb();
  
  walldist = geometry->node[iPoint]->GetWall_Distance();

  su2double gradu[3][3], gradT[3], gradk[3], gradomega[3], divu, taut[3][3], tautomut[3][3],
            delta[3][3] = {{1.0, 0.0, 0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}},
            pk = 0, pw = 0;

  const su2double F1 = varTur->GetF1blending(iPoint);
  const su2double F2 = varTur->GetF2blending(iPoint);

  const su2double alfa        = F1*constants[8] + (1.0 - F1)*constants[9];
  const su2double sigmak      = F1*constants[0] + (1.0 - F1)*constants[1];
  const su2double sigmaomega  = F1*constants[2] + (1.0 - F1)*constants[3];
  const su2double sigmaomega2 = constants[3];
  const su2double beta        = F1*constants[4] + (1.0 - F1)*constants[5];
  const su2double betastar    = constants[6];
  const su2double a1          = constants[7];
  const su2double CDkw        = varTur->GetCrossDiff(iPoint);

  const su2double VorticityMag = varFlo->GetVorticityMag(iPoint);

  const bool stress_limited = (omega < VorticityMag*F2/a1);
  const su2double zeta = max(omega,VorticityMag*F2/a1);

  for (iDim = 0; iDim < nDim; iDim++) {
    for (jDim = 0 ; jDim < nDim; jDim++) {
      gradu[iDim][jDim] = varFlo->GetGradient_Primitive(iPoint, iDim+1, jDim);
    }
    gradT[iDim]     = varFlo->GetGradient_Primitive(iPoint, 0, iDim);
    gradk[iDim]     = varTur->GetGradient(iPoint, 0, iDim);
    gradomega[iDim] = varTur->GetGradient(iPoint, 1, iDim);
  }
  
  //--- Account for wall functions
  su2double wf = varFlo->GetTauWallFactor(iPoint);

  divu = 0.0; for (iDim = 0 ; iDim < nDim; ++iDim) divu += gradu[iDim][iDim];

  for (iDim = 0; iDim < nDim; ++iDim) {
    for (jDim = 0; jDim < nDim; ++jDim) {
      taut[iDim][jDim] = wf*(mut*( gradu[jDim][iDim] + gradu[iDim][jDim] )
                       - TWO3*mut*divu*delta[iDim][jDim]
                       - TWO3*r*k*delta[iDim][jDim]);
      tautomut[iDim][jDim] = wf*( gradu[jDim][iDim] + gradu[iDim][jDim] 
                           - TWO3*divu*delta[iDim][jDim]
                           - TWO3*zeta*delta[iDim][jDim]);
      pk += 1./wf*taut[iDim][jDim]*gradu[iDim][jDim];
      pw += 1./wf*tautomut[iDim][jDim]*gradu[iDim][jDim];
    }
  }

  const bool pk_limited    = (pk > 20.*betastar*r*omega*k);
  const bool pk_positive   = (pk >= 0);
  const bool pw_positive   = (pw >= 0);
  const bool cdkw_positive = (!varTur->GetCrossDiffLimited(iPoint));

  //--- Momentum weights
  vector<su2double> TmpWeights(weights[0].size(), 0.0);
  su2double factor = 0.0;
  for (iDim = 0; iDim < nDim; ++iDim) {
    factor = 0.0;
    factor += -TWO3*divu*alfa*varAdjTur->GetGradient_Adaptation(iPoint, 1, iDim)*pw_positive;
    if (!pk_limited) {   
      factor += -TWO3*divu*mut/r*varAdjTur->GetGradient_Adaptation(iPoint, 0, iDim)*pk_positive;
    }
    for (jDim = 0; jDim < nDim; ++jDim) {
      factor += (tautomut[iDim][jDim]+(gradu[iDim][jDim]+gradu[jDim][iDim]))*alfa*varAdjTur->GetGradient_Adaptation(iPoint, 1, jDim)*pw_positive;
      if (!pk_limited) {
         factor += (taut[iDim][jDim]+mut*(gradu[iDim][jDim]+gradu[jDim][iDim]))/r*varAdjTur->GetGradient_Adaptation(iPoint, 0, jDim)*pk_positive;
      }
    }
    TmpWeights[iDim+1] += factor;
  }

  //--- k and omega weights
  factor = 0.0;
  for (iDim = 0; iDim < nDim; ++iDim) {
    for (jDim = 0; jDim < nDim; ++jDim) {
      iVar = iDim+1;
      factor += (tautomut[iDim][jDim]+wf*TWO3*zeta*delta[iDim][jDim])
              * (varAdjFlo->GetGradient_Adaptation(iPoint, iVar, jDim)
              + u[jDim]*varAdjFlo->GetGradient_Adaptation(iPoint, (nVarFlo-1), iDim));
    }
    factor += cp/Prt*gradT[iDim]*varAdjFlo->GetGradient_Adaptation(iPoint, (nVarFlo-1), iDim);
    factor += sigmak*gradk[iDim]*varAdjTur->GetGradient_Adaptation(iPoint, 0, iDim)
            + 1.0/sigmak*gradk[iDim]*varAdjFlo->GetGradient_Adaptation(iPoint, (nVarFlo-1), iDim)
            + sigmaomega*gradomega[iDim]*varAdjTur->GetGradient_Adaptation(iPoint, 1, iDim);
  }

  TmpWeights[nVarFlo+0] += factor/zeta;
  TmpWeights[nVarFlo+1] += -k*factor/pow(zeta,2.)*(!stress_limited);
  if (cdkw_positive) {
    for (iDim = 0; iDim < nDim; ++iDim) {
      TmpWeights[nVarFlo+0] += 2.*(1.-F1)*sigmaomega2/omega*gradomega[iDim]*varAdjTur->GetGradient_Adaptation(iPoint, 1, iDim);
      TmpWeights[nVarFlo+1] += 2.*(1.-F1)*sigmaomega2/omega*gradk[iDim]*varAdjTur->GetGradient_Adaptation(iPoint, 1, iDim);
    }
  }

  //--- Density weight
  for (iDim = 0; iDim < nDim; ++iDim) TmpWeights[0] += -u[iDim]*TmpWeights[iDim+1];
  TmpWeights[0] += -k*TmpWeights[nVarFlo+0] - omega*TmpWeights[nVarFlo+1]
                 + k/zeta*factor*(!stress_limited);

  //--- Add TmpWeights to weights, then reset for second-order terms
  for (iVar = 0; iVar < nVarFlo+nVarTur; ++iVar) weights[1][iVar] += TmpWeights[iVar];
  fill(TmpWeights.begin(), TmpWeights.end(), 0.0);

  //--- Second-order terms (error due to gradients)
  if(nDim == 3) {
    const unsigned short rki = 0, romegai = 1, rei = (nVarFlo - 1), xxi = 0, yyi = 3, zzi = 5;
    TmpWeights[nVarFlo+0] += -(mu+sigmak*mut)/r*(varAdjTur->GetHessian(iPoint, rki, xxi)
                                                +varAdjTur->GetHessian(iPoint, rki, yyi)
                                                +varAdjTur->GetHessian(iPoint, rki, zzi))
                             -(mu+mut/sigmak)/r*(varAdjFlo->GetHessian(iPoint, rei, xxi)
                                                +varAdjFlo->GetHessian(iPoint, rei, yyi)
                                                +varAdjFlo->GetHessian(iPoint, rei, zzi)); // Hk
    TmpWeights[nVarFlo+1] += -(mu+sigmaomega*mut)/r*(varAdjTur->GetHessian(iPoint, romegai, xxi)
                                                    +varAdjTur->GetHessian(iPoint, romegai, yyi)
                                                    +varAdjTur->GetHessian(iPoint, romegai, zzi)); // Homega

  }
  else {
    const unsigned short rki = 0, romegai = 1, rei = (nVarFlo - 1), xxi = 0, yyi = 2;
    TmpWeights[nVarFlo+0] += -(mu+sigmak*mut)/r*(varAdjTur->GetHessian(iPoint, rki, xxi)
                                                +varAdjTur->GetHessian(iPoint, rki, yyi))
                             -(mu+mut/sigmak)/r*(varAdjFlo->GetHessian(iPoint, rei, xxi)
                                                +varAdjFlo->GetHessian(iPoint, rei, yyi)); // Hk
    TmpWeights[nVarFlo+1] += -(mu+sigmaomega*mut)/r*(varAdjTur->GetHessian(iPoint, romegai, xxi)
                                                    +varAdjTur->GetHessian(iPoint, romegai, yyi)); // Homega
  }
  TmpWeights[0] += -k*TmpWeights[nVarFlo+0]-omega*TmpWeights[nVarFlo+1];

  //--- Add TmpWeights to weights
  weights[2][0]         += TmpWeights[0];
  weights[2][nVarFlo+0] += TmpWeights[nVarFlo+0];
  weights[2][nVarFlo+1] += TmpWeights[nVarFlo+1];

  //--- Zeroth-order terms due to production
  if (!pk_limited){
    weights[0][nVarFlo+0] += TWO3*divu*varAdjTur->GetSolution(iPoint,0)*pk_positive;
  }
  else {
    weights[0][0]         += 20.*betastar*k*omega*varAdjTur->GetSolution(iPoint,0);
    weights[0][nVarFlo+0] += -20.*betastar*omega*varAdjTur->GetSolution(iPoint,0);
    weights[0][nVarFlo+1] += -20.*betastar*k*varAdjTur->GetSolution(iPoint,0);
  }
  weights[0][nVarFlo+1] += TWO3*(!stress_limited)*alfa*divu*varAdjTur->GetSolution(iPoint,1)*pw_positive;
  
  //--- Zeroth-order terms due to dissipation
  weights[0][0]         += -betastar*k*omega*varAdjTur->GetSolution(iPoint,0)
                         - beta*pow(omega,2.)*varAdjTur->GetSolution(iPoint,1);
  weights[0][nVarFlo+0] += betastar*omega*varAdjTur->GetSolution(iPoint,0);
  weights[0][nVarFlo+1] += betastar*k*varAdjTur->GetSolution(iPoint,0)
                         + 2.*beta*omega*varAdjTur->GetSolution(iPoint,1);
  
  //--- Zeroth-order terms due to cross-diffusion
  // weights[0][nVarFlo+1] += (1. - F1)*CDkw/(r*zeta)*varAdjTur->GetSolution(iPoint,1)*(!stress_limited)*(cdkw_positive);
  weights[0][nVarFlo+1] += (1. - F1)*CDkw/(r*omega)*varAdjTur->GetSolution(iPoint,1)*cdkw_positive;

}
