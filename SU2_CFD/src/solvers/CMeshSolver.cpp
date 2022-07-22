/*!
 * \file CMeshSolver.cpp
 * \brief Main subroutines to solve moving meshes using a pseudo-linear elastic approach.
 * \author Ruben Sanchez
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

#include "../../../Common/include/adt/CADTPointsOnlyClass.hpp"
#include "../../../Common/include/parallelization/omp_structure.hpp"
#include "../../include/solvers/CMeshSolver.hpp"
#include "../../../Common/include/toolboxes/geometry_toolbox.hpp"

using namespace GeometryToolbox;


CMeshSolver::CMeshSolver(CGeometry *geometry, CConfig *config) : CFEASolver(LINEAR_SOLVER_MODE::MESH_DEFORM) {

  /*--- Initialize some booleans that determine the kind of problem at hand. ---*/

  time_domain = config->GetTime_Domain();
  multizone = config->GetMultizone_Problem();

  /*--- Determine if the stiffness per-element is set ---*/
  switch (config->GetDeform_Stiffness_Type()) {
  case INVERSE_VOLUME:
  case SOLID_WALL_DISTANCE:
    stiffness_set = false;
    break;
  case CONSTANT_STIFFNESS:
    stiffness_set = true;
    break;
  }

  /*--- Initialize the number of spatial dimensions, length of the state
   vector (same as spatial dimensions for grid deformation), and grid nodes. ---*/

  unsigned short iDim;
  unsigned long iPoint, iElem;

  nDim         = geometry->GetnDim();
  nVar         = geometry->GetnDim();
  nPoint       = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();
  nElement     = geometry->GetnElem();

  MinVolume_Ref = 0.0;
  MinVolume_Curr = 0.0;

  MaxVolume_Ref = 0.0;
  MaxVolume_Curr = 0.0;

  /*--- Initialize the node structure ---*/

  nodes = new CMeshBoundVariable(nPoint, nDim, config);
  SetBaseClassPointerToNodes();

  /*--- Set which points are vertices and allocate boundary data. ---*/

  for (iPoint = 0; iPoint < nPoint; iPoint++) {

    for (iDim = 0; iDim < nDim; ++iDim)
      nodes->SetMesh_Coord(iPoint, iDim, geometry->nodes->GetCoord(iPoint, iDim));

    for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
      long iVertex = geometry->nodes->GetVertex(iPoint, iMarker);
      if (iVertex >= 0) {
        nodes->Set_isVertex(iPoint,true);
        break;
      }
    }
  }
  static_cast<CMeshBoundVariable*>(nodes)->AllocateBoundaryVariables(config);

  if (config->GetAeroelastic_Modal() || config->GetImposed_Modal_Move()) SetStructuralModes(geometry, config);
  
  /*--- Initialize the element structure ---*/

  element.resize(nElement);

  /*--- Initialize matrix, solution, and r.h.s. structures for the linear solver. ---*/

  if (rank == MASTER_NODE) cout << "Initialize Jacobian structure (Mesh Deformation)." << endl;

  LinSysSol.Initialize(nPoint, nPointDomain, nVar, 0.0);
  LinSysRes.Initialize(nPoint, nPointDomain, nVar, 0.0);
  Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, false, geometry, config);

  /*--- Initialize structures for hybrid-parallel mode. ---*/

  HybridParallelInitialization(geometry);

  /*--- Element container structure. ---*/

  if (nDim == 2) {
    for(int thread = 0; thread < omp_get_max_threads(); ++thread) {

      const int offset = thread*MAX_FE_KINDS;
      element_container[FEA_TERM][EL_TRIA+offset] = new CTRIA1();
      element_container[FEA_TERM][EL_QUAD+offset] = new CQUAD4();
    }
  }
  else {
    for(int thread = 0; thread < omp_get_max_threads(); ++thread) {

      const int offset = thread*MAX_FE_KINDS;
      element_container[FEA_TERM][EL_TETRA+offset] = new CTETRA1();
      element_container[FEA_TERM][EL_HEXA +offset] = new CHEXA8 ();
      element_container[FEA_TERM][EL_PYRAM+offset] = new CPYRAM5();
      element_container[FEA_TERM][EL_PRISM+offset] = new CPRISM6();
    }
  }

  /*--- Initialize the BGS residuals in multizone problems. ---*/
  if (config->GetMultizone_Residual()){
    Residual_BGS.resize(nVar,0.0);
    Residual_Max_BGS.resize(nVar,0.0);
    Point_Max_BGS.resize(nVar,0);
    Point_Max_Coord_BGS.resize(nVar,nDim) = su2double(0.0);
  }


  /*--- Allocate element properties - only the index, to allow further integration with CFEASolver on a later stage ---*/
  element_properties = new CProperty*[nElement];
  for (iElem = 0; iElem < nElement; iElem++){
    element_properties[iElem] = new CProperty(iElem);
  }

  /*--- Compute the element volumes using the reference coordinates ---*/
  SU2_OMP_PARALLEL {
    SetMinMaxVolume(geometry, config, false);
  }
  END_SU2_OMP_PARALLEL

  /*--- Compute the wall distance using the reference coordinates ---*/
  SetWallDistance(geometry, config);

  if (size != SINGLE_NODE) {
    vector<unsigned short> essentialMarkers;
    /*--- Markers types covered in SetBoundaryDisplacements. ---*/
    for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
      if (((config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE) &&
           (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) ||
           (config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
           (config->GetMarker_All_Moving(iMarker) == YES)) {
        essentialMarkers.push_back(iMarker);
      }
    }
    Set_VertexEliminationSchedule(geometry, essentialMarkers);
  }
}

void CMeshSolver::SetMinMaxVolume(CGeometry *geometry, CConfig *config, bool updated) {

  /*--- This routine is for post processing, it does not need to be recorded. ---*/
  const bool wasActive = AD::BeginPassive();

  /*--- Initialize shared reduction variables. ---*/
  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
    MaxVolume = -1E22; MinVolume = 1E22;
    ElemCounter = 0;
  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS

  /*--- Local min/max, final reduction outside loop. ---*/
  su2double maxVol = -1E22, minVol = 1E22;
  unsigned long elCount = 0;

  /*--- Loop over the elements in the domain. ---*/

  SU2_OMP_FOR_DYN(omp_chunk_size)
  for (unsigned long iElem = 0; iElem < nElement; iElem++) {

    int thread = omp_get_thread_num();

    int EL_KIND;
    unsigned short iNode, nNodes, iDim;

    GetElemKindAndNumNodes(geometry->elem[iElem]->GetVTK_Type(), EL_KIND, nNodes);

    CElement* fea_elem = element_container[FEA_TERM][EL_KIND + thread*MAX_FE_KINDS];

    /*--- For the number of nodes, we get the coordinates from
     *    the connectivity matrix and the geometry structure. ---*/

    for (iNode = 0; iNode < nNodes; iNode++) {

      auto indexNode = geometry->elem[iElem]->GetNode(iNode);

      /*--- Compute the volume with the reference or current coordinates. ---*/
      for (iDim = 0; iDim < nDim; iDim++) {
        su2double val_Coord = nodes->GetMesh_Coord(indexNode,iDim);
        if (updated)
          val_Coord += nodes->GetSolution(indexNode,iDim);

        fea_elem->SetRef_Coord(iNode, iDim, val_Coord);
      }
    }

    /*--- Compute the volume of the element (or the area in 2D cases ). ---*/

    su2double ElemVolume;
    if (nDim == 2) ElemVolume = fea_elem->ComputeArea();
    else           ElemVolume = fea_elem->ComputeVolume();

    maxVol = max(maxVol, ElemVolume);
    minVol = min(minVol, ElemVolume);

    if (updated) element[iElem].SetCurr_Volume(ElemVolume);
    else element[iElem].SetRef_Volume(ElemVolume);

    /*--- Count distorted elements. ---*/
    if (ElemVolume <= 0.0) elCount++;
  }
  END_SU2_OMP_FOR
  SU2_OMP_CRITICAL
  {
    MaxVolume = max(MaxVolume, maxVol);
    MinVolume = min(MinVolume, minVol);
    ElemCounter += elCount;
  }
  END_SU2_OMP_CRITICAL

  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
  {
    elCount = ElemCounter; maxVol = MaxVolume; minVol = MinVolume;
    SU2_MPI::Allreduce(&elCount, &ElemCounter, 1, MPI_UNSIGNED_LONG, MPI_SUM, SU2_MPI::GetComm());
    SU2_MPI::Allreduce(&maxVol, &MaxVolume, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
    SU2_MPI::Allreduce(&minVol, &MinVolume, 1, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS

  /*--- Volume from 0 to 1 ---*/

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iElem = 0; iElem < nElement; iElem++) {
    if (updated) {
      su2double ElemVolume = element[iElem].GetCurr_Volume()/MaxVolume;
      element[iElem].SetCurr_Volume(ElemVolume);
    }
    else {
      su2double ElemVolume = element[iElem].GetRef_Volume()/MaxVolume;
      element[iElem].SetRef_Volume(ElemVolume);
    }
  }
  END_SU2_OMP_FOR

  /*--- Store the maximum and minimum volume. ---*/
  BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS {
  if (updated) {
    MaxVolume_Curr = MaxVolume;
    MinVolume_Curr = MinVolume;
  }
  else {
    MaxVolume_Ref = MaxVolume;
    MinVolume_Ref = MinVolume;
  }

  if ((ElemCounter != 0) && (rank == MASTER_NODE))
    cout <<"There are " << ElemCounter << " elements with negative volume.\n" << endl;

  }
  END_SU2_OMP_SAFE_GLOBAL_ACCESS

  AD::EndPassive(wasActive);

}

void CMeshSolver::SetWallDistance(CGeometry *geometry, CConfig *config) {

  /*--- Initialize min and max distance ---*/

  MaxDistance = -1E22; MinDistance = 1E22;

  /*--- Compute the total number of nodes on no-slip boundaries ---*/

  unsigned long nVertex_SolidWall = 0;
  for(auto iMarker=0u; iMarker<config->GetnMarker_All(); ++iMarker) {
    if(config->GetSolid_Wall(iMarker) && !config->GetMarker_All_Deform_Mesh_Sym_Plane(iMarker)) {
      nVertex_SolidWall += geometry->GetnVertex(iMarker);
    }
  }

  /*--- Allocate the vectors to hold boundary node coordinates
   and its local ID. ---*/

  vector<su2double>     Coord_bound(nDim*nVertex_SolidWall);
  vector<unsigned long> PointIDs(nVertex_SolidWall);

  /*--- Retrieve and store the coordinates of the no-slip boundary nodes
   and their local point IDs. ---*/


  for (unsigned long iMarker=0, ii=0, jj=0; iMarker<config->GetnMarker_All(); ++iMarker) {

    if (!config->GetSolid_Wall(iMarker) || config->GetMarker_All_Deform_Mesh_Sym_Plane(iMarker)) continue;

    for (auto iVertex=0u; iVertex<geometry->GetnVertex(iMarker); ++iVertex) {
      auto iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
      PointIDs[jj++] = iPoint;
      for (auto iDim=0u; iDim<nDim; ++iDim){
        Coord_bound[ii++] = nodes->GetMesh_Coord(iPoint,iDim);
      }
    }
  }

  /*--- Build the ADT of the boundary nodes. ---*/

  CADTPointsOnlyClass WallADT(nDim, nVertex_SolidWall, Coord_bound.data(),
                              PointIDs.data(), true);

  SU2_OMP_PARALLEL
  {
  /*--- Loop over all interior mesh nodes and compute the distances to each
   of the no-slip boundary nodes. Store the minimum distance to the wall
   for each interior mesh node. ---*/

  if( WallADT.IsEmpty() ) {

    /*--- No solid wall boundary nodes in the entire mesh. Set the
     wall distance to MaxDistance so we get stiffness of 1. ---*/

    SU2_OMP_FOR_STAT(omp_chunk_size)
    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      nodes->SetWallDistance(iPoint, MaxDistance);
    }
    END_SU2_OMP_FOR
  }
  else {
    su2double MaxDistance_Local = -1E22, MinDistance_Local = 1E22;

    /*--- Solid wall boundary nodes are present. Compute the wall
     distance for all nodes. ---*/
    SU2_OMP_FOR_DYN(omp_chunk_size)
    for(auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      su2double dist;
      unsigned long pointID;
      int rankID;
      WallADT.DetermineNearestNode(nodes->GetMesh_Coord(iPoint), dist,
                                   pointID, rankID);
      nodes->SetWallDistance(iPoint,dist);

      MaxDistance_Local = max(MaxDistance_Local, dist);

      /*--- To discard points on the surface we use > EPS ---*/

      if (dist > EPS)  MinDistance_Local = min(MinDistance_Local, dist);

    }
    END_SU2_OMP_FOR
    SU2_OMP_CRITICAL
    {
      MaxDistance = max(MaxDistance, MaxDistance_Local);
      MinDistance = min(MinDistance, MinDistance_Local);
    }
    END_SU2_OMP_CRITICAL

    BEGIN_SU2_OMP_SAFE_GLOBAL_ACCESS
    {
      MaxDistance_Local = MaxDistance;
      MinDistance_Local = MinDistance;
      SU2_MPI::Allreduce(&MaxDistance_Local, &MaxDistance, 1, MPI_DOUBLE, MPI_MAX, SU2_MPI::GetComm());
      SU2_MPI::Allreduce(&MinDistance_Local, &MinDistance, 1, MPI_DOUBLE, MPI_MIN, SU2_MPI::GetComm());
    }
    END_SU2_OMP_SAFE_GLOBAL_ACCESS
  }

  /*--- Normalize distance from 0 to 1 ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
    su2double nodeDist = nodes->GetWallDistance(iPoint)/MaxDistance;
    nodes->SetWallDistance(iPoint,nodeDist);
  }
  END_SU2_OMP_FOR

  /*--- Compute the element distances ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (auto iElem = 0ul; iElem < nElement; iElem++) {

    int EL_KIND;
    unsigned short nNodes = 0;
    GetElemKindAndNumNodes(geometry->elem[iElem]->GetVTK_Type(), EL_KIND, nNodes);

    /*--- Average the distance of the nodes in the element ---*/

    su2double ElemDist = 0.0;
    for (auto iNode = 0u; iNode < nNodes; iNode++) {
      auto iPoint = geometry->elem[iElem]->GetNode(iNode);
      ElemDist += nodes->GetWallDistance(iPoint);
    }
    ElemDist = ElemDist/su2double(nNodes);

    element[iElem].SetWallDistance(ElemDist);
  }
  END_SU2_OMP_FOR

  }
  END_SU2_OMP_PARALLEL
}

void CMeshSolver::SetMesh_Stiffness(CGeometry **geometry, CNumerics **numerics, CConfig *config){

  if (stiffness_set) return;

  /*--- Use the config option as an upper bound on elasticity modulus.
   *    For RANS meshes the range of element volume or wall distance is
   *    very large and leads to an ill-conditioned stiffness matrix.
   *    Absolute values of elasticity modulus are not important for
   *    mesh deformation, since linear elasticity is used and all
   *    boundary conditions are essential (Dirichlet). ---*/
  const su2double maxE = config->GetDeform_ElasticityMod();

  /*--- All threads must execute the entire loop (no worksharing),
   *    each sets the stiffnesses for its numerics instance. ---*/
  SU2_OMP_PARALLEL
  {
  CNumerics* myNumerics = numerics[FEA_TERM + omp_get_thread_num()*MAX_TERMS];

  switch (config->GetDeform_Stiffness_Type()) {

    /*--- Stiffness inverse of the volume of the element. ---*/
    case INVERSE_VOLUME:
      for (unsigned long iElem = 0; iElem < nElement; iElem++) {
        su2double E = 1.0 / element[iElem].GetRef_Volume();
        myNumerics->SetMeshElasticProperties(iElem, min(E,maxE));
      }
    break;

    /*--- Stiffness inverse of the distance of the element to the closest wall. ---*/
    case SOLID_WALL_DISTANCE: {
      const su2double offset = config->GetDeform_StiffLayerSize();
      if (fabs(offset) > 0.0) {
        /*--- With prescribed layer of maximum stiffness (reaches max and holds). ---*/
        su2double d0 = offset / MaxDistance;
        su2double dmin = 1.0 / maxE;
        su2double scale = 1.0 / (1.0 - d0);
        for (unsigned long iElem = 0; iElem < nElement; iElem++) {
          su2double E = 1.0 / max(dmin, (element[iElem].GetWallDistance() - d0)*scale);
          myNumerics->SetMeshElasticProperties(iElem, E);
        }
      } else {
        /*--- Without prescribed layer of maximum stiffness (may not reach max). ---*/
        for (unsigned long iElem = 0; iElem < nElement; iElem++) {
          su2double E = 1.0 / element[iElem].GetWallDistance();
          myNumerics->SetMeshElasticProperties(iElem, min(E,maxE));
        }
      }
    }
    break;
  }
  }
  END_SU2_OMP_PARALLEL

  stiffness_set = true;

}

void CMeshSolver::DeformMesh(CGeometry **geometry, CNumerics **numerics, CConfig *config){

  if (multizone) nodes->Set_BGSSolution_k();

  /*--- Capture a few MPI dependencies for AD. ---*/
  geometry[MESH_0]->InitiateComms(geometry[MESH_0], config, COORDINATES);
  geometry[MESH_0]->CompleteComms(geometry[MESH_0], config, COORDINATES);

  InitiateComms(geometry[MESH_0], config, SOLUTION);
  CompleteComms(geometry[MESH_0], config, SOLUTION);

  InitiateComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);
  CompleteComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);

  /*--- Compute the stiffness matrix, no point recording because we clear the residual. ---*/

  const bool wasActive = AD::BeginPassive();

  Compute_StiffMatrix(geometry[MESH_0], numerics, config);

  AD::EndPassive(wasActive);

  /*--- Clear residual (loses AD info), we do not want an incremental solution. ---*/
  SU2_OMP_PARALLEL {
    LinSysRes.SetValZero();

    if (time_domain && config->GetFSI_Simulation()) {
      SU2_OMP_FOR_STAT(omp_chunk_size)
      for (unsigned long iPoint = 0; iPoint < nPoint; ++iPoint)
        for (unsigned short iDim = 0; iDim < nDim; ++iDim)
          LinSysSol(iPoint, iDim) = nodes->GetSolution(iPoint, iDim);
      END_SU2_OMP_FOR
    }
  }
  END_SU2_OMP_PARALLEL

  /*--- Impose boundary conditions (all of them are ESSENTIAL BC's - displacements). ---*/
  SetBoundaryDisplacements(geometry[MESH_0], config, false);

  /*--- Solve the linear system. ---*/
  Solve_System(geometry[MESH_0], config);

  SU2_OMP_PARALLEL {

  /*--- Update the grid coordinates and cell volumes using the solution
     of the linear system (usol contains the x, y, z displacements). ---*/
  UpdateGridCoord(geometry[MESH_0], config);

  /*--- Update the dual grid. ---*/
  CGeometry::UpdateGeometry(geometry, config);

  /*--- Check for failed deformation (negative volumes). ---*/
  SetMinMaxVolume(geometry[MESH_0], config, true);

  /*--- The Grid Velocity is only computed if the problem is time domain ---*/
  if (time_domain && !config->GetFSI_Simulation())
    ComputeGridVelocity(geometry, config);

  }
  END_SU2_OMP_PARALLEL

  if (time_domain && config->GetFSI_Simulation()) {
    ComputeGridVelocity_FromBoundary(geometry, numerics, config);
  }

}

void CMeshSolver::DeformMeshHB(CGeometry **geometry, CNumerics **numerics, CConfig *config, 
		             unsigned long TimeIter){

  if (multizone) nodes->Set_BGSSolution_k();

  /*--- Capture a few MPI dependencies for AD. ---*/
  geometry[MESH_0]->InitiateComms(geometry[MESH_0], config, COORDINATES);
  geometry[MESH_0]->CompleteComms(geometry[MESH_0], config, COORDINATES);

  InitiateComms(geometry[MESH_0], config, SOLUTION);
  CompleteComms(geometry[MESH_0], config, SOLUTION);

  InitiateComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);
  CompleteComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);

  /*--- Compute the stiffness matrix, no point recording because we clear the residual. ---*/

  const bool wasActive = AD::BeginPassive();

  Compute_StiffMatrix(geometry[MESH_0], numerics, config);

  AD::EndPassive(wasActive);

  /*--- Clear residual (loses AD info), we do not want an incremental solution. ---*/
  SU2_OMP_PARALLEL {
    LinSysRes.SetValZero();
  }

  /*--- Impose boundary conditions (all of them are ESSENTIAL BC's - displacements). ---*/
  SetBoundaryDisplacementsHB(geometry[MESH_0], numerics[FEA_TERM], TimeIter, config);
  
  /*--- Solve the linear system. ---*/
  Solve_System(geometry[MESH_0], config);
  
  SU2_OMP_PARALLEL {

  /*--- Update the grid coordinates and cell volumes using the solution
     of the linear system (usol contains the x, y, z displacements). ---*/
  UpdateGridCoord(geometry[MESH_0], config);

  /*--- Update the dual grid. ---*/
  UpdateDualGrid(geometry[MESH_0], config);

  /*--- The Grid Velocity is only computed if the problem is time domain ---*/
  if (config->GetBnd_Velo()){
  ComputeGridVelocity_FromBoundary(geometry, numerics, config); 
  }
  /*--- Update the multigrid structure. ---*/
  UpdateMultiGrid(geometry, config);

  /*--- Check for failed deformation (negative volumes). ---*/
  SetMinMaxVolume(geometry[MESH_0], config, true);

  } // end parallel

}

void CMeshSolver::AeroelasticDeformMesh(CGeometry **geometry, CNumerics **numerics, CConfig *config, su2double* structural_solution, unsigned long iter) {

  vector<su2double> str_sol(4,0.0);
  cout << "Disp at AeroDeformMesh = " ;
  for (int i=0;i<4;i++) {
	  str_sol[i] = structural_solution[i];

	  cout << str_sol[i] << " ";
  }
  cout << endl;

    Surface_Aeroelastic(geometry[MESH_0], config, str_sol, iter);

  if (multizone) nodes->Set_BGSSolution_k();

  /*--- Capture a few MPI dependencies for AD. ---*/
  geometry[MESH_0]->InitiateComms(geometry[MESH_0], config, COORDINATES);
  geometry[MESH_0]->CompleteComms(geometry[MESH_0], config, COORDINATES);

  InitiateComms(geometry[MESH_0], config, SOLUTION);
  CompleteComms(geometry[MESH_0], config, SOLUTION);

  InitiateComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);
  CompleteComms(geometry[MESH_0], config, MESH_DISPLACEMENTS);

  /*--- Compute the stiffness matrix, no point recording because we clear the residual. ---*/

  const bool wasActive = AD::BeginPassive();

  Compute_StiffMatrix(geometry[MESH_0], numerics, config);

  AD::EndPassive(wasActive);

  /*--- Clear residual (loses AD info), we do not want an incremental solution. ---*/
  SU2_OMP_PARALLEL {
    LinSysRes.SetValZero();
  }

  /*--- Impose boundary conditions (all of them are ESSENTIAL BC's - displacements). ---*/
  SetBoundaryDisplacements(geometry[MESH_0], config, false);
 
//  cout << endl << "boundary set." << endl;

 
  /*--- Solve the linear system. ---*/
  Solve_System(geometry[MESH_0], config);
 
//  cout << endl << "system solved." << endl;

 
  SU2_OMP_PARALLEL {

  /*--- Update the grid coordinates and cell volumes using the solution
     of the linear system (usol contains the x, y, z displacements). ---*/
  UpdateGridCoord(geometry[MESH_0], config);

  /*--- Update the dual grid. ---*/
  UpdateDualGrid(geometry[MESH_0], config);

//  cout << endl << "grid updated." << endl;


  /*--- The Grid Velocity is only computed if the problem is time domain ---*/
 // ComputeGridVelocity_FromBoundary(geometry, numerics, config); 
   //ComputeGridVelocity(geometry[MESH_0], config);
  //CompareGridVelocity(geometry[MESH_0], config);
  /*--- Update the multigrid structure. ---*/
  UpdateMultiGrid(geometry, config);

  /*--- Check for failed deformation (negative volumes). ---*/
  SetMinMaxVolume(geometry[MESH_0], config, true);

  } // end parallel

}

void CMeshSolver::UpdateGridCoord(CGeometry *geometry, const CConfig *config){

  /*--- Update the grid coordinates using the solution of the linear system ---*/

  /*--- LinSysSol contains the absolute x, y, z displacements. ---*/
  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++){
    for (unsigned short iDim = 0; iDim < nDim; iDim++) {
      /*--- Retrieve the displacement from the solution of the linear system ---*/
      su2double val_disp = LinSysSol(iPoint, iDim);
      /*--- Store the displacement of the mesh node ---*/
      nodes->SetSolution(iPoint, iDim, val_disp);
      /*--- Compute the current coordinate as Mesh_Coord + Displacement ---*/
      su2double val_coord = nodes->GetMesh_Coord(iPoint,iDim) + val_disp;
      /*--- Update the geometry container ---*/
      geometry->nodes->SetCoord(iPoint, iDim, val_coord);
    }
  }
  END_SU2_OMP_FOR

  /*--- Communicate the updated displacements and mesh coordinates. ---*/
  geometry->InitiateComms(geometry, config, COORDINATES);
  geometry->CompleteComms(geometry, config, COORDINATES);

}

void CMeshSolver::UpdateDualGrid(CGeometry *geometry, CConfig *config){

  /*--- After moving all nodes, update the dual mesh. Recompute the edges and
   dual mesh control volumes in the domain and on the boundaries. ---*/

  geometry->SetControlVolume(config, UPDATE);
  geometry->SetBoundControlVolume(config, UPDATE);
  geometry->SetMaxLength(config);

}

void CMeshSolver::ComputeGridVelocity_FromBoundary(CGeometry **geometry, CNumerics **numerics, CConfig *config){

  if (config->GetnZone() == 1)
    SU2_MPI::Error("It is not possible to compute grid velocity from boundary velocity for single zone problems.\n"
                   "MARKER_FLUID_LOAD should only be used for structural boundaries.", CURRENT_FUNCTION);

  /*--- Compute the stiffness matrix, no point recording because we clear the residual. ---*/

  const bool wasActive = AD::BeginPassive();

  Compute_StiffMatrix(geometry[MESH_0], numerics, config);

  AD::EndPassive(wasActive);

  const su2double velRef = config->GetVelocity_Ref();
  const su2double invVelRef = 1.0 / velRef;

  /*--- Clear residual (loses AD info), we do not want an incremental solution. ---*/
  SU2_OMP_PARALLEL {
    LinSysRes.SetValZero();

    SU2_OMP_FOR_STAT(omp_chunk_size)
    for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++)
      for (unsigned short iDim = 0; iDim < nDim; iDim++)
        LinSysSol(iPoint, iDim) = geometry[MESH_0]->nodes->GetGridVel(iPoint)[iDim] * velRef;
    END_SU2_OMP_FOR
  }
  END_SU2_OMP_PARALLEL

  /*--- Impose boundary conditions including boundary velocity ---*/
  SetBoundaryDisplacements(geometry[MESH_0], config, true);

  /*--- Solve the linear system. ---*/
  Solve_System(geometry[MESH_0], config);

  SU2_OMP_PARALLEL {
    SU2_OMP_FOR_STAT(omp_chunk_size)
    for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++)
      for (unsigned short iDim = 0; iDim < nDim; iDim++)
        geometry[MESH_0]->nodes->SetGridVel(iPoint, iDim, LinSysSol(iPoint,iDim)*invVelRef);
    END_SU2_OMP_FOR

    for (auto iMGlevel = 1u; iMGlevel <= config->GetnMGLevels(); iMGlevel++)
      geometry[iMGlevel]->SetRestricted_GridVelocity(geometry[iMGlevel-1]);
  }
  END_SU2_OMP_PARALLEL

}

void CMeshSolver::UpdateMultiGrid(CGeometry **geometry, CConfig *config) const{

  /*--- Update the multigrid structure after moving the finest grid,
   including computing the grid velocities on the coarser levels
   when the problem is solved in unsteady conditions. ---*/

  for (auto iMGlevel = 1u; iMGlevel <= config->GetnMGLevels(); iMGlevel++) {
    const auto iMGfine = iMGlevel-1;
    // geometry[iMGlevel]->SetControlVolume(config, geometry[iMGfine], UPDATE);
    // geometry[iMGlevel]->SetBoundControlVolume(config, geometry[iMGfine],UPDATE);
    geometry[iMGlevel]->SetCoord(geometry[iMGfine]);
    if (time_domain)
      geometry[iMGlevel]->SetRestricted_GridVelocity(geometry[iMGfine]);
  }

}

void CMeshSolver::ComputeGridVelocity(CGeometry **geometry, const CConfig *config) const {

  /*--- Compute the velocity of each node. ---*/

  const bool firstOrder = config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_1ST;
  const bool secondOrder = config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND;
  const su2double invTimeStep = 1.0 / config->GetDelta_UnstTimeND();

  SU2_OMP_FOR_STAT(omp_chunk_size)
  for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++) {

    /*--- Coordinates of the current point at n+1, n, & n-1 time levels. ---*/

    const su2double* Disp_nM1 = nodes->GetSolution_time_n1(iPoint);
    const su2double* Disp_n   = nodes->GetSolution_time_n(iPoint);
    const su2double* Disp_nP1 = nodes->GetSolution(iPoint);

    /*--- Compute mesh velocity for this point with 1st or 2nd-order approximation. ---*/

    for (unsigned short iDim = 0; iDim < nDim; iDim++) {

      su2double GridVel = 0.0;
      if (firstOrder)
        GridVel = (Disp_nP1[iDim] - Disp_n[iDim]) * invTimeStep;
      else if (secondOrder)
        GridVel = (1.5*Disp_nP1[iDim] - 2.0*Disp_n[iDim] + 0.5*Disp_nM1[iDim]) * invTimeStep;

      geometry[MESH_0]->nodes->SetGridVel(iPoint, iDim, GridVel);
    }
  }
  END_SU2_OMP_FOR

  for (auto iMGlevel = 1u; iMGlevel <= config->GetnMGLevels(); iMGlevel++)
    geometry[iMGlevel]->SetRestricted_GridVelocity(geometry[iMGlevel-1]);

}

void CMeshSolver::BC_Deforming(CGeometry *geometry, const CConfig *config, unsigned short val_marker, bool velocity){

  for (auto iVertex = 0ul; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    const auto iPoint = geometry->vertex[val_marker][iVertex]->GetNode();

    /*--- Retrieve the boundary displacement or velocity ---*/
    su2double Sol[MAXNVAR] = {0.0};
    for (unsigned short iDim = 0; iDim < nDim; iDim++) {
      if (velocity) Sol[iDim] = nodes->GetBound_Vel(iPoint,iDim);
      else Sol[iDim] = nodes->GetBound_Disp(iPoint,iDim);
    }
    LinSysSol.SetBlock(iPoint, Sol);
    Jacobian.EnforceSolutionAtNode(iPoint, Sol, LinSysRes);
  }
}

void CMeshSolver::SetBoundaryDisplacements(CGeometry *geometry, CConfig *config, bool velocity_transfer){

  /* Surface motions are not applied during discrete adjoint runs as the corresponding
   * boundary displacements are computed when loading the primal solution, and it
   * would be complex to account for the incremental nature of these motions.
   * The derivatives are still correct since the motion does not depend on the solution,
   * but this means that (for now) we cannot get derivatives w.r.t. motion parameters. */

  // if (rank == MASTER_NODE) cout << "Set Boundary Conditions:" << endl;
  // string Marker_Tag, Moving_Tag;

  if (config->GetSurface_Movement(DEFORMING) && !config->GetDiscrete_Adjoint() && !config->GetAeroelastic_Modal() && !config->GetAeroelasticity_HB()) {
    if (velocity_transfer)
      SU2_MPI::Error("Forced motions are not compatible with FSI simulations.", CURRENT_FUNCTION);

    if (rank == MASTER_NODE)
      cout << endl << " Updating surface positions." << endl;

    Surface_Translating(geometry, config, config->GetTimeIter());
    Surface_Plunging(geometry, config, config->GetTimeIter());
    Surface_Pitching(geometry, config, config->GetTimeIter());
    Surface_Rotating(geometry, config, config->GetTimeIter());
  }

  unsigned short iMarker;

  /*--- Impose zero displacements of all non-moving surfaces that are not MARKER_DEFORM_SYM_PLANE. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Deform_Mesh_Sym_Plane(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE)) {

      BC_Clamped(geometry, config, iMarker);
    }
  }

  /*--- Impose displacement boundary conditions and symmetry. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

      BC_Deforming(geometry, config, iMarker, velocity_transfer);
    }
    else if (config->GetMarker_All_Deform_Mesh_Sym_Plane(iMarker) == YES) {

      BC_Sym_Plane(geometry, config, iMarker);
    }
  }

  /*--- Clamp far away nodes according to deform limit. ---*/
  if ((config->GetDeform_Stiffness_Type() == SOLID_WALL_DISTANCE) &&
      (config->GetDeform_Limit() < MaxDistance)) {

    const su2double limit = config->GetDeform_Limit() / MaxDistance;

    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      if (nodes->GetWallDistance(iPoint) <= limit) continue;

      su2double zeros[MAXNVAR] = {0.0};
      nodes->SetSolution(iPoint, zeros);
      LinSysSol.SetBlock(iPoint, zeros);
      Jacobian.EnforceSolutionAtNode(iPoint, zeros, LinSysRes);
    }
  }

  /*--- Clamp nodes outside of a given area. ---*/
  if (config->GetHold_GridFixed()) {

    auto MinCoordValues = config->GetHold_GridFixed_Coord();
    auto MaxCoordValues = &config->GetHold_GridFixed_Coord()[3];

    for (auto iPoint = 0ul; iPoint < geometry->GetnPoint(); iPoint++) {
      auto Coord = geometry->nodes->GetCoord(iPoint);
      for (auto iDim = 0; iDim < nDim; iDim++) {
        if ((Coord[iDim] < MinCoordValues[iDim]) || (Coord[iDim] > MaxCoordValues[iDim])) {
          su2double zeros[MAXNVAR] = {0.0};
          nodes->SetSolution(iPoint, zeros);
          LinSysSol.SetBlock(iPoint, zeros);
          Jacobian.EnforceSolutionAtNode(iPoint, zeros, LinSysRes);
          break;
        }
      }
    }
  }

}

void CMeshSolver::SetBoundaryVelocities(CGeometry *geometry, CNumerics *numerics, CConfig *config){

  unsigned short iMarker;

  /*--- Impose zero displacements of all non-moving surfaces (also at nodes in multiple moving/non-moving boundaries). ---*/
  /*--- Exceptions: symmetry plane, the receive boundaries and periodic boundaries should get a different treatment. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) != SYMMETRY_PLANE) &&
        (config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE) &&
        (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {

      BC_Clamped(geometry, config, iMarker);
    }
  }

  /*--- Symmetry plane is clamped, for now. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE)) {

      BC_Clamped(geometry, config, iMarker);
    }
  }

  /*--- Impose velocity boundary conditions. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

	    cout << "DEFORMING" << endl;
	    
            cout << "Marker= " << config->GetMarker_All_TagBound(iMarker) << endl;

      BC_Velocity(geometry, numerics, config, iMarker);
    }
  }

  /*--- Clamp far away nodes according to deform limit. ---*/
  if ((config->GetDeform_Stiffness_Type() == SOLID_WALL_DISTANCE) &&
      (config->GetDeform_Limit() < MaxDistance)) {

    const su2double limit = config->GetDeform_Limit() / MaxDistance;

    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      if (nodes->GetWallDistance(iPoint) <= limit) continue;

      su2double zeros[MAXNVAR] = {0.0};
      nodes->SetSolution(iPoint, zeros);
      LinSysSol.SetBlock(iPoint, zeros);
      Jacobian.EnforceSolutionAtNode(iPoint, zeros, LinSysRes);
    }
  }
}

void CMeshSolver::SetBoundaryDisplacementsHB(CGeometry *geometry, CNumerics *numerics, unsigned long TimeIter, CConfig *config){

  /* Surface motions are not applied during discrete adjoint runs as the corresponding
   * boundary displacements are computed when loading the primal solution, and it
   * would be complex to account for the incremental nature of these motions.
   * The derivatives are still correct since the motion does not depend on the solution,
   * but this means that (for now) we cannot get derivatives w.r.t. motion parameters. */

  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);

  if (config->GetSurface_Movement(DEFORMING) && !config->GetDiscrete_Adjoint() && !config->GetAeroelastic_Modal() && !config->GetImposed_Modal_Move()) {

    if (rank == MASTER_NODE)
      cout << endl << " Updating surface positions HB." << endl;

    //Surface_Translating(geometry, config, TimeIter);
    Surface_Plunging(geometry, config, TimeIter);
    Surface_Pitching(geometry, config, TimeIter);
    //Surface_Rotating(geometry, config, TimeIter); 

  }

  unsigned short iMarker;

  /*--- Impose zero displacements of all non-moving surfaces (also at nodes in multiple moving/non-moving boundaries). ---*/
  /*--- Exceptions: symmetry plane, the receive boundaries and periodic boundaries should get a different treatment. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) != SYMMETRY_PLANE) &&
        (config->GetMarker_All_KindBC(iMarker) != SEND_RECEIVE) &&
        (config->GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY)) {

      BC_Clamped(geometry, config, iMarker);
    }
  }

  /*--- Symmetry plane is clamped, for now. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == NO) &&
        (config->GetMarker_All_Moving(iMarker) == NO) &&
        (config->GetMarker_All_KindBC(iMarker) == SYMMETRY_PLANE)) {

      BC_Clamped(geometry, config, iMarker);
    }
  }

  /*--- Impose displacement boundary conditions. ---*/
  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

      BC_Deforming(geometry, config, iMarker, false);
    }
  }

  /*--- Clamp far away nodes according to deform limit. ---*/
  if ((config->GetDeform_Stiffness_Type() == SOLID_WALL_DISTANCE) &&
      (config->GetDeform_Limit() < MaxDistance)) {

    const su2double limit = config->GetDeform_Limit() / MaxDistance;

    for (auto iPoint = 0ul; iPoint < nPoint; ++iPoint) {
      if (nodes->GetWallDistance(iPoint) <= limit) continue;

      su2double zeros[MAXNVAR] = {0.0};
      nodes->SetSolution(iPoint, zeros);
      LinSysSol.SetBlock(iPoint, zeros);
      Jacobian.EnforceSolutionAtNode(iPoint, zeros, LinSysRes);
    }
  }

}

void CMeshSolver::SetDualTime_Mesh(void){

  nodes->Set_Solution_time_n1();
  nodes->Set_Solution_time_n();
}

void CMeshSolver::LoadRestart(CGeometry **geometry, CSolver ***solver, CConfig *config, int val_iter, bool val_update_geo) {

  /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

  string filename = config->GetFilename(config->GetSolution_FileName(), "", val_iter);

  if (config->GetRead_Binary_Restart()) {
    Read_SU2_Restart_Binary(geometry[MESH_0], config, filename);
  } else {
    Read_SU2_Restart_ASCII(geometry[MESH_0], config, filename);
  }

  /*--- Load data from the restart into correct containers. ---*/

  unsigned long iPoint_Global, counter = 0;

  for (iPoint_Global = 0; iPoint_Global < geometry[MESH_0]->GetGlobal_nPointDomain(); iPoint_Global++) {

    /*--- Retrieve local index. If this node from the restart file lives
     on the current processor, we will load and instantiate the vars. ---*/

    auto iPoint_Local = geometry[MESH_0]->GetGlobal_to_Local_Point(iPoint_Global);

    if (iPoint_Local >= 0) {

      /*--- We need to store this point's data, so jump to the correct
       offset in the buffer of data from the restart file and load it. ---*/

      auto index = counter*Restart_Vars[1];

      for (unsigned short iDim = 0; iDim < nDim; iDim++){
        /*--- Update the coordinates of the mesh ---*/
        su2double curr_coord = Restart_Data[index+iDim];
        /// TODO: "Double deformation" in multizone adjoint if this is set here?
        ///       In any case it should not be needed as deformation is called before other solvers
        ///geometry[MESH_0]->nodes->SetCoord(iPoint_Local, iDim, curr_coord);

        /*--- Store the displacements computed as the current coordinates
         minus the coordinates of the reference mesh file ---*/
        su2double displ = curr_coord - nodes->GetMesh_Coord(iPoint_Local, iDim);
        nodes->SetSolution(iPoint_Local, iDim, displ);
      }

      /*--- Increment the overall counter for how many points have been loaded. ---*/
      counter++;
    }

  }

  /*--- Detect a wrong solution file ---*/

  if (counter != nPointDomain) {
    SU2_MPI::Error(string("The solution file ") + filename + string(" doesn't match with the mesh file!\n") +
                   string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
  }

  /*--- Communicate the loaded displacements. ---*/
  solver[MESH_0][MESH_SOL]->InitiateComms(geometry[MESH_0], config, SOLUTION);
  solver[MESH_0][MESH_SOL]->CompleteComms(geometry[MESH_0], config, SOLUTION);

  /*--- Init the linear system solution. ---*/
  for (unsigned long iPoint = 0; iPoint < nPoint; ++iPoint) {
    for (unsigned short iDim = 0; iDim < nDim; ++iDim) {
      LinSysSol(iPoint, iDim) = nodes->GetSolution(iPoint, iDim);
    }
  }

  /*--- For time-domain problems, we need to compute the grid velocities. ---*/
  if (time_domain && !config->GetFSI_Simulation()) {
    /*--- Update the old geometry (coordinates n and n-1) ---*/
    RestartOldGeometry(geometry[MESH_0], config);

    /*--- Once Displacement_n and Displacement_n1 are filled we can compute the Grid Velocity ---*/
    ComputeGridVelocity(geometry, config);
  }

  /*--- Store the boundary displacements at the Bound_Disp variable. ---*/

  for (unsigned short iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {

    if ((config->GetMarker_All_Deform_Mesh(iMarker) == YES) ||
        (config->GetMarker_All_Moving(iMarker) == YES)) {

      for (unsigned long iVertex = 0; iVertex < geometry[MESH_0]->nVertex[iMarker]; iVertex++) {

        /*--- Get node index. ---*/
        auto iNode = geometry[MESH_0]->vertex[iMarker][iVertex]->GetNode();

        /*--- Set boundary solution. ---*/
        nodes->SetBound_Disp(iNode, nodes->GetSolution(iNode));
      }
    }
  }

  /*--- Delete the class memory that is used to load the restart. ---*/

  delete [] Restart_Vars; Restart_Vars = nullptr;
  delete [] Restart_Data; Restart_Data = nullptr;

}

void CMeshSolver::RestartOldGeometry(CGeometry *geometry, const CConfig *config) {

  /*--- This function is intended for dual time simulations ---*/

  unsigned short iZone = config->GetiZone();
  unsigned short nZone = geometry->GetnZone();
  string filename = config->GetSolution_FileName();

  /*--- Multizone problems require the number of the zone to be appended. ---*/

  if (nZone > 1)
    filename = config->GetMultizone_FileName(filename, iZone, "");

  /*--- Determine how many files need to be read. ---*/

  unsigned short nSteps = (config->GetTime_Marching() == TIME_MARCHING::DT_STEPPING_2ND) ? 2 : 1;

  for(unsigned short iStep = 1; iStep <= nSteps; ++iStep) {

    unsigned short CommType = (iStep == 1) ? SOLUTION_TIME_N : SOLUTION_TIME_N1;

    /*--- Modify file name for an unsteady restart ---*/
    int Unst_RestartIter;
    if (!config->GetDiscrete_Adjoint())
      Unst_RestartIter = static_cast<int>(config->GetRestart_Iter()) - iStep;
    else
      Unst_RestartIter = static_cast<int>(config->GetUnst_AdjointIter()) - config->GetTimeIter() - iStep - 1;

    if (Unst_RestartIter < 0) {

      if (rank == MASTER_NODE) cout << "Requested mesh restart filename is negative. Setting zero displacement" << endl;

      for (unsigned long iPoint = 0; iPoint < nPoint; ++iPoint) {
        for (unsigned short iDim = 0; iDim < nDim; iDim++) {
          if(iStep == 1) nodes->Set_Solution_time_n(iPoint, iDim, 0.0);
          else nodes->Set_Solution_time_n1(iPoint, iDim, 0.0);
        }
      }
    }
    else {
      string filename_n = config->GetUnsteady_FileName(filename, Unst_RestartIter, "");

      /*--- Read the restart data from either an ASCII or binary SU2 file. ---*/

      if (config->GetRead_Binary_Restart()) {
        Read_SU2_Restart_Binary(geometry, config, filename_n);
      } else {
        Read_SU2_Restart_ASCII(geometry, config, filename_n);
      }

      /*--- Load data from the restart into correct containers. ---*/

      unsigned long iPoint_Global, counter = 0;

      for (iPoint_Global = 0; iPoint_Global < geometry->GetGlobal_nPointDomain(); iPoint_Global++) {

        /*--- Retrieve local index. If this node from the restart file lives
         on the current processor, we will load and instantiate the vars. ---*/

        auto iPoint_Local = geometry->GetGlobal_to_Local_Point(iPoint_Global);

        if (iPoint_Local >= 0) {

          /*--- We need to store this point's data, so jump to the correct
           offset in the buffer of data from the restart file and load it. ---*/

          auto index = counter*Restart_Vars[1];

          for (unsigned short iDim = 0; iDim < nDim; iDim++) {
            su2double curr_coord = Restart_Data[index+iDim];
            su2double displ = curr_coord - nodes->GetMesh_Coord(iPoint_Local,iDim);

            if(iStep==1)
              nodes->Set_Solution_time_n(iPoint_Local, iDim, displ);
            else
              nodes->Set_Solution_time_n1(iPoint_Local, iDim, displ);
          }

          /*--- Increment the overall counter for how many points have been loaded. ---*/
          counter++;
        }
      }


      /*--- Detect a wrong solution file. ---*/

      if (counter != nPointDomain) {
        SU2_MPI::Error(string("The solution file ") + filename_n + string(" doesn't match with the mesh file!\n") +
                       string("It could be empty lines at the end of the file."), CURRENT_FUNCTION);
      }
    }

    /*--- Delete the class memory that is used to load the restart. ---*/

    delete [] Restart_Vars; Restart_Vars = nullptr;
    delete [] Restart_Data; Restart_Data = nullptr;

    InitiateComms(geometry, config, CommType);
    CompleteComms(geometry, config, CommType);

  } // iStep

}

void CMeshSolver::Surface_Pitching(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  const su2double* Coord = nullptr;
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, VelCoord[3] = {0.0}, Omega[3] = {0.0}, Ampl[3] = {0.0}, Phase[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  su2double alphaDot[3], newGridVel[3] = {0.0,0.0,0.0};
  su2double rotCoord[3] = {0.0}, r[3] = {0.0};
  su2double rotMatrix[3][3] = {{0.0}};
  su2double dtheta, dphi, dpsi;
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker, iDim;
  unsigned long iPoint, iVertex;
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  string Marker_Tag, Moving_Tag;

  bool hbaero  = config->GetAeroelasticity_HB();
  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  /*--- Compute delta time based on physical time step ---*/
  if (harmonic_balance) {
    /*--- period of oscillation & time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period/TimeInstances;
  }
  

  time_new = iter*deltaT;
  if (harmonic_balance) {
      /*--- For harmonic balance, begin movement from the zero position ---*/
      time_old = 0.0;
  }
  else {
  if (iter == 0) time_old = time_new;
  else time_old = (iter-1)*deltaT;
  }
  /*--- Store displacement of each node on the pitching surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to pitch ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Pitching origin, frequency, and amplitude from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Ampl[iDim]   = config->GetMarkerPitching_Ampl(jMarker, iDim)*DEG2RAD;
        Omega[iDim]  = config->GetMarkerPitching_Omega(jMarker, iDim)/config->GetOmega_Ref();
        Phase[iDim]  = config->GetMarkerPitching_Phase(jMarker, iDim)*DEG2RAD;
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }
      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing pitching displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Pitching frequency: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s about origin: (" << Center[0];
          cout << ", " << Center[1] << ", " << Center[2] << ")." << endl;
          cout << " Pitching amplitude about origin: (" << Ampl[0]/DEG2RAD;
          cout << ", " << Ampl[1]/DEG2RAD << ", " << Ampl[2]/DEG2RAD;
          cout << ") degrees."<< endl;
          cout << " Pitching phase lag about origin: (" << Phase[0]/DEG2RAD;
          cout << ", " << Phase[1]/DEG2RAD <<", "<< Phase[2]/DEG2RAD;
          cout << ") degrees."<< endl;
        }
      }

      /*--- Compute delta change in the angle about the x, y, & z axes. ---*/

      dtheta = -Ampl[0]*(sin(Omega[0]*time_new + Phase[0])
                       - sin(Omega[0]*time_old + Phase[0]));
      dphi   = -Ampl[1]*(sin(Omega[1]*time_new + Phase[1])
                       - sin(Omega[1]*time_old + Phase[1]));
      dpsi   = -Ampl[2]*(sin(Omega[2]*time_new + Phase[2])
                       - sin(Omega[2]*time_old + Phase[2]));

      if (hbaero) config->SetHB_pitch(-dpsi,iter);

      /*--- Angular velocity at the new time ---*/

      alphaDot[0] = -Omega[0]*Ampl[0]*cos(Omega[0]*time_new);
      alphaDot[1] = -Omega[1]*Ampl[1]*cos(Omega[1]*time_new);
      alphaDot[2] = -Omega[2]*Ampl[2]*cos(Omega[2]*time_new);

      if (hbaero) config->SetHB_pitch_rate(-alphaDot[2],iter);

      /*--- Compute rotation matrix. ---*/

      RotationMatrix(dtheta, dphi, dpsi, rotMatrix);

      /*--- Apply rotation to the vertices. ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Index and coordinates of the current point ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        Coord  = geometry->nodes->GetCoord(iPoint);
        //GridVel = geometry->nodes->GetGridVel(iPoint);

        /*--- Calculate non-dim. position from rotation center ---*/

        for (iDim = 0; iDim < nDim; iDim++)
          r[iDim] = (Coord[iDim]-Center[iDim])/Lref;

        /*--- Compute transformed point coordinates ---*/

    
     	Rotate(rotMatrix, Center, r, rotCoord);

     
    	/*--- Cross Product of angular velocity and distance from center.
    	 Note that we have assumed the grid velocities have been set to
         an initial value in the plunging routine. ---*/
       

        /*--- Calculate delta change in the x, y, & z directions ---*/
        for (iDim = 0; iDim < nDim; iDim++){
          VarCoord[iDim] = (rotCoord[iDim]-Coord[iDim])/Lref;
          VelCoord[iDim] = (rotCoord[iDim]-Center[iDim])/Lref;
	}
	
	newGridVel[0] =  alphaDot[1]*VelCoord[2] - alphaDot[2]*VelCoord[1];
        newGridVel[1] =  alphaDot[2]*VelCoord[0] - alphaDot[0]*VelCoord[2];
        if (nDim == 3) newGridVel[2] = alphaDot[0]*VelCoord[1] - alphaDot[1]*VelCoord[0];

//	newGridVel[0] =  alphaDot[1]*rotCoord[2] - alphaDot[2]*rotCoord[1];
  //      newGridVel[1] =  alphaDot[2]*rotCoord[0] - alphaDot[0]*rotCoord[2];
    //    if (nDim == 3) newGridVel[2] = alphaDot[0]*rotCoord[1] - alphaDot[1]*rotCoord[0];

        /*--- Set node displacement for volume deformation ---*/	

        for (iDim = 0; iDim < nDim; iDim++){
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];
          newGridVel[iDim] += nodes->GetBound_Vel(iPoint, iDim);
  //         geometry->nodes->SetGridVel(iPoint, iDim, newGridVel[iDim]);
	}

        nodes->SetBound_Disp(iPoint, VarCoordAbs);

        //if (harmonic_balance) 
	nodes->SetBound_Vel(iPoint, newGridVel);
      }
    }
  }
  /*--- For pitching we don't update the motion origin and moment reference origin. ---*/

}

void CMeshSolver::Surface_Rotating(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  const su2double* Coord = nullptr;
  su2double VarCoordAbs[3] = {0.0};
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, VelCoord[3] = {0.0},
	    Omega[3] = {0.0}, rotCoord[3] = {0.0}, r[3] = {0.0}, 
	    Center_Aux[3] = {0.0};
  su2double newGridVel[3] = {0.0,0.0,0.0}; 
  su2double rotMatrix[3][3] = {{0.0}};
  su2double dtheta, dphi, dpsi;
  unsigned short iMarker, jMarker, iDim;
  unsigned long iPoint, iVertex;
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  string Marker_Tag, Moving_Tag;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  /*-- Set dt for harmonic balance cases ---*/
  if (harmonic_balance) {
    /*--- period of oscillation & compute time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period /TimeInstances;
  }

  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (harmonic_balance) {
      /*--- For harmonic balance, begin movement from the zero position ---*/
      time_old = 0.0;
  } else {
      time_old = time_new;
      if (iter != 0) time_old = (iter-1)*deltaT;
  }
  /*--- Store displacement of each node on the rotating surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to rotate ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Rotation origin and angular velocity from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Omega[iDim]  = config->GetMarkerRotationRate(jMarker, iDim)/config->GetOmega_Ref();
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing rotating displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Angular velocity: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s about origin: (" << Center[0];
          cout << ", " << Center[1] << ", " << Center[2] << ")." << endl;
        }
      }

      /*--- Compute delta change in the angle about the x, y, & z axes. ---*/

      dtheta = Omega[0]*(time_new-time_old);
      dphi   = Omega[1]*(time_new-time_old);
      dpsi   = Omega[2]*(time_new-time_old);

      /*--- Compute rotation matrix. ---*/

      RotationMatrix(dtheta, dphi, dpsi, rotMatrix);

      /*--- Apply rotation to the vertices. ---*/

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Index and coordinates of the current point ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        Coord  = geometry->nodes->GetCoord(iPoint);
        //GridVel = geometry->nodes->GetGridVel(iPoint);

        /*--- Calculate non-dim. position from rotation center ---*/

        for (iDim = 0; iDim < nDim; iDim++)
          r[iDim] = (Coord[iDim]-Center[iDim])/Lref;

        /*--- Compute transformed point coordinates ---*/

        Rotate(rotMatrix, Center, r, rotCoord);

	/*--- Calculate delta change in the x, y, & z directions ---*/
        for (iDim = 0; iDim < nDim; iDim++){
          VarCoord[iDim] = (rotCoord[iDim]-Coord[iDim])/Lref;
	  VelCoord[iDim] = (rotCoord[iDim]-Center[iDim])/Lref;
	}

	newGridVel[0] = Omega[1]*VelCoord[2] - Omega[2]*VelCoord[1];
	newGridVel[1] = Omega[2]*VelCoord[0] - Omega[0]*VelCoord[2];
        if (nDim == 3)newGridVel[2] = Omega[0]*VelCoord[1] - Omega[1]*VelCoord[0];


        /*--- Set node displacement for volume deformation ---*/
        for (iDim = 0; iDim < nDim; iDim++){
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];
	  newGridVel[iDim] += nodes->GetBound_Vel(iPoint, iDim); 
         // if (harmonic_balance) geometry->nodes->SetGridVel(iPoint, iDim, newGridVel[iDim]);
        }

        nodes->SetBound_Disp(iPoint, VarCoordAbs);

        //if (harmonic_balance)
	nodes->SetBound_Vel(iPoint, newGridVel);
      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
   same rotation movement, because we use the last markers rotation matrix and center ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the rotation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker < config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (config->GetMoveMotion_Origin(jMarker) != YES) continue;

    for (iDim = 0; iDim < 3; iDim++)
      Center_Aux[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);

    /*--- Calculate non-dim. position from rotation center ---*/

    for (iDim = 0; iDim < nDim; iDim++)
      r[iDim] = (Center_Aux[iDim]-Center[iDim])/Lref;

    /*--- Compute transformed point coordinates ---*/

    Rotate(rotMatrix, Center, r, rotCoord);

    /*--- Calculate delta change in the x, y, & z directions ---*/
    for (iDim = 0; iDim < nDim; iDim++)
      VarCoord[iDim] = (rotCoord[iDim]-Center_Aux[iDim])/Lref;

    for (iDim = 0; iDim < 3; iDim++)
      Center_Aux[iDim] += VarCoord[iDim];

    config->SetMarkerMotion_Origin(Center_Aux, jMarker);
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the rotation. ---*/

  for (jMarker=0; jMarker<config->GetnMarker_Monitoring(); jMarker++) {

    Center_Aux[0] = config->GetRefOriginMoment_X(jMarker);
    Center_Aux[1] = config->GetRefOriginMoment_Y(jMarker);
    Center_Aux[2] = config->GetRefOriginMoment_Z(jMarker);

    /*--- Calculate non-dim. position from rotation center ---*/

    for (iDim = 0; iDim < nDim; iDim++)
      r[iDim] = (Center_Aux[iDim]-Center[iDim])/Lref;

    /*--- Compute transformed point coordinates ---*/

    Rotate(rotMatrix, Center, r, rotCoord);

    /*--- Calculate delta change in the x, y, & z directions ---*/
    for (iDim = 0; iDim < nDim; iDim++)
      VarCoord[iDim] = (rotCoord[iDim]-Center_Aux[iDim])/Lref;

    config->SetRefOriginMoment_X(jMarker, Center_Aux[0]+VarCoord[0]);
    config->SetRefOriginMoment_Y(jMarker, Center_Aux[1]+VarCoord[1]);
    config->SetRefOriginMoment_Z(jMarker, Center_Aux[2]+VarCoord[2]);
  }
}

void CMeshSolver::Surface_Plunging(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  su2double Center[3] = {0.0}, VarCoord[3] = {0.0}, Omega[3] = {0.0}, Ampl[3] = {0.0}, Phase[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  su2double newGridVel[3] = {0.0, 0.0, 0.0}, xDot[3];
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  unsigned short iDim;

  bool hbaero  = config->GetAeroelasticity_HB();
  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  Lref   = config->GetLength_Ref();

  su2double b = Lref/2;
  /*--- Compute delta time based on physical time step ---*/

    if (harmonic_balance) {
    /*--- period of oscillation & time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period/TimeInstances;
  }

  time_new = iter*deltaT;
  if (harmonic_balance) {
      /*--- For harmonic balance, begin movement from the zero position ---*/
      time_old = 0.0;
    } else {
      time_old = time_new;
      if (iter != 0) time_old = (iter-1)*deltaT;
    }
  

  /*--- Store displacement of each node on the plunging surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to plunge ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      /*--- Plunging frequency and amplitude from config. ---*/

      for (iDim = 0; iDim < 3; iDim++){
        Ampl[iDim]   = config->GetMarkerPlunging_Ampl(jMarker, iDim)/Lref;
        Omega[iDim]  = config->GetMarkerPlunging_Omega(jMarker, iDim)/config->GetOmega_Ref();
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
        Phase[iDim]  = config->GetMarkerPlunging_Phase(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Physical Time = " << time_new << endl;
        cout << " Storing plunging displacement for marker: ";
        cout << Marker_Tag << "." << endl;
        if (iter == 0) {
          cout << " Plunging frequency: (" << Omega[0] << ", " << Omega[1];
          cout << ", " << Omega[2] << ") rad/s." << endl;
          cout << " Plunging amplitude: (" << Ampl[0]/DEG2RAD;
          cout << ", " << Ampl[1]/DEG2RAD << ", " << Ampl[2]/DEG2RAD;
          cout << ") degrees."<< endl;
        }
      }

      /*--- Compute delta change in the position in the x, y, & z directions. ---*/

      VarCoord[0] = -Ampl[0]*(sin(Omega[0]*time_new + Phase[0]) - sin(Omega[0]*time_old));
      VarCoord[1] = -Ampl[1]*(sin(Omega[1]*time_new + Phase[1]) - sin(Omega[1]*time_old));
      VarCoord[2] = -Ampl[2]*(sin(Omega[2]*time_new + Phase[2]) - sin(Omega[2]*time_old));

      if (hbaero)  config->SetHB_plunge(-VarCoord[1]/b,iter);

      /*--- Compute grid velocity due to plunge in the x, y, & z directions. ---*/
      xDot[0] = -Ampl[0]*Omega[0]*(cos(Omega[0]*time_new + Phase[0]));
      xDot[1] = -Ampl[1]*Omega[1]*(cos(Omega[1]*time_new + Phase[1]));
      xDot[2] = -Ampl[2]*Omega[2]*(cos(Omega[2]*time_new + Phase[2]));

      if (hbaero) config->SetHB_plunge_rate(-xDot[1],iter);

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

     	newGridVel[0] = xDot[0];
	newGridVel[1] = xDot[1];
     	if (nDim == 3) newGridVel[2] = xDot[2];

        for (iDim = 0; iDim < nDim; iDim++) {
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];
	  newGridVel[iDim] += nodes->GetBound_Vel(iPoint, iDim);
        }

        nodes->SetBound_Disp(iPoint, VarCoordAbs);
	nodes->SetBound_Vel(iPoint, newGridVel);

      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
   same plunging movement, because we use the last VarCoord set ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the translation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker<config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (!harmonic_balance) {
    if (config->GetMoveMotion_Origin(jMarker) == YES) {
      for (iDim = 0; iDim < 3; iDim++)
        Center[iDim] += VarCoord[iDim];

      config->SetMarkerMotion_Origin(Center, jMarker);
    }
    }
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the plunging. ---*/

  if (!harmonic_balance) {
  for (jMarker=0; jMarker < config->GetnMarker_Monitoring(); jMarker++) {
    Center[0] = config->GetRefOriginMoment_X(jMarker) + VarCoord[0];
    Center[1] = config->GetRefOriginMoment_Y(jMarker) + VarCoord[1];
    Center[2] = config->GetRefOriginMoment_Z(jMarker) + VarCoord[2];
    config->SetRefOriginMoment_X(jMarker, Center[0]);
    config->SetRefOriginMoment_Y(jMarker, Center[1]);
    config->SetRefOriginMoment_Z(jMarker, Center[2]);
  }
  }
  else {
    Center[0] = config->GetRefOriginMoment_X(0) + VarCoord[0];
    Center[1] = config->GetRefOriginMoment_Y(0) + VarCoord[1];
    Center[2] = config->GetRefOriginMoment_Z(0) + VarCoord[2];
    config->SetRefOriginMoment_X_HB(iter, Center[0]);
    config->SetRefOriginMoment_Y_HB(iter, Center[1]);
    config->SetRefOriginMoment_Z_HB(iter, Center[2]);
  }

}

void CMeshSolver::Surface_Translating(CGeometry *geometry, CConfig *config, unsigned long iter) {

  su2double deltaT, time_new, time_old;
  su2double Center[3] = {0.0}, newGridVel[3] = {0.0}, VarCoord[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0};
  su2double xDot[3] = {0.0};
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  unsigned short iDim;

  /*--- Retrieve values from the config file ---*/

  deltaT = config->GetDelta_UnstTimeND();
  if (harmonic_balance) {
    /*--- period of oscillation & time interval using nTimeInstances ---*/
    su2double period = config->GetHarmonicBalance_Period();
    period /= config->GetTime_Ref();
    su2double TimeInstances = config->GetnTimeInstances();
    deltaT = period/TimeInstances;
  }
  /*--- Compute delta time based on physical time step ---*/

  time_new = iter*deltaT;
  if (harmonic_balance) {
      /*--- For harmonic balance, begin movement from the zero position ---*/
      time_old = 0.0;  
  } else {
      time_old = time_new;
      if (iter != 0) time_old = (iter-1.0)*deltaT;  
  }  
  /*--- Store displacement of each node on the translating surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to translate ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

      for (iDim = 0; iDim < 3; iDim++) {
        xDot[iDim]   = config->GetMarkerTranslationRate(jMarker, iDim);
        Center[iDim] = config->GetMarkerMotion_Origin(jMarker, iDim);
      }

      /*--- Print some information to the console. Be verbose at the first
       iteration only (mostly for debugging purposes). ---*/
      // Note that the MASTER_NODE might not contain all the markers being moved.

      if (rank == MASTER_NODE) {
        cout << " Storing translating displacement for marker: ";
        cout << Marker_Tag << "." << endl;
	cout << " Physical Time = " << time_new << endl;
        if (iter == 0) {
          cout << " Translational velocity: (" << xDot[0]*config->GetVelocity_Ref() << ", " << xDot[1]*config->GetVelocity_Ref();
          cout << ", " << xDot[2]*config->GetVelocity_Ref();
          if (config->GetSystemMeasurements() == SI) cout << ") m/s." << endl;
          else cout << ") ft/s." << endl;
        }
      }

      /*--- Compute delta change in the position in the x, y, & z directions. ---*/

      VarCoord[0] = xDot[0]*(time_new-time_old);
      VarCoord[1] = xDot[1]*(time_new-time_old);
      VarCoord[2] = xDot[2]*(time_new-time_old);

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

        for (iDim = 0; iDim < nDim; iDim++){
          VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + VarCoord[iDim];
	  newGridVel[iDim]  = xDot[iDim];
        }

        nodes->SetBound_Disp(iPoint, VarCoordAbs);
	nodes->SetBound_Vel(iPoint, newGridVel);
      }
    }
  }

  /*--- When updating the origins it is assumed that all markers have the
        same translational velocity, because we use the last VarCoord set ---*/

  /*--- Set the mesh motion center to the new location after
   incrementing the position with the translation. This new
   location will be used for subsequent mesh motion for the given marker.---*/

  for (jMarker=0; jMarker < config->GetnMarker_Moving(); jMarker++) {

    /*-- Check if we want to update the motion origin for the given marker ---*/

    if (config->GetMoveMotion_Origin(jMarker) == YES) {
      for (iDim = 0; iDim < 3; iDim++)
        Center[iDim] += VarCoord[iDim];

      config->SetMarkerMotion_Origin(Center, jMarker);
    }
  }

  /*--- Set the moment computation center to the new location after
   incrementing the position with the translation. ---*/

  for (jMarker=0; jMarker < config->GetnMarker_Monitoring(); jMarker++) {
    Center[0] = config->GetRefOriginMoment_X(jMarker) + VarCoord[0];
    Center[1] = config->GetRefOriginMoment_Y(jMarker) + VarCoord[1];
    Center[2] = config->GetRefOriginMoment_Z(jMarker) + VarCoord[2];
    config->SetRefOriginMoment_X(jMarker, Center[0]);
    config->SetRefOriginMoment_Y(jMarker, Center[1]);
    config->SetRefOriginMoment_Z(jMarker, Center[2]);
  }
}

void CMeshSolver::Surface_Aeroelastic(CGeometry *geometry, CConfig *config, vector<su2double> &structural_solution, unsigned long iter) {

  su2double deltaT, time_new, time_old, Lref;
  const su2double* Coord = nullptr;
  su2double Center[3] = {0.0}, CenterOrg[3] = {0.0}, VarCoord[3] = {0.0}, VelCoord[3] = {0.0}, dU[3] = {0.0};
  su2double VarCoordAbs[3] = {0.0}, xDot[3];
  su2double alphaDot[3], newGridVel[3] = {0.0,0.0,0.0};
  su2double rotCoord[3] = {0.0}, r[3] = {0.0};
  su2double rotMatrix[3][3] = {{0.0}};
  su2double dtheta, dphi, dpsi;
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker, iDim;
  unsigned long iPoint, iVertex;
  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  string Marker_Tag, Moving_Tag;

  Lref   = config->GetLength_Ref();

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

     // if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
       // continue;
      //}

      if (rank == MASTER_NODE) {
        cout << " Storing plunging & Pitching displacement for marker: ";
        cout << Marker_Tag << "." << endl; 
      }

      cout<< "STR SOL"<<endl;
      for (int i=0;i<4;i++){
      cout << structural_solution[i] << " ";
      }
      cout << endl;
      /*--- Compute delta change in the position in the x, y, & z directions. ---*/

      dU[0] = 0.0;
      dU[1] = -structural_solution[0];
      dU[2] = 0.0;

      /*--- Compute grid velocity due to plunge in the x, y, & z directions. ---*/
      xDot[0] = 0.0;
      xDot[1] = -structural_solution[2];
      xDot[2] = 0.0;

      /*--- Compute delta change in the angle about the x, y, & z axes. ---*/
 
      dtheta = 0.0; 
      dphi   = 0.0;     
      dpsi   = -structural_solution[1];

      /*--- Angular velocity at the new time ---*/
      alphaDot[0] = 0.0;
      alphaDot[1] = 0.0;
      alphaDot[2] = -structural_solution[3];
      /*--- Compute rotation matrix. ---*/

      RotationMatrix(dtheta, dphi, dpsi, rotMatrix);

      /*--- Apply rotation to the vertices. ---*/


 //   if (config->GetMoveMotion_Origin(jMarker) == YES) {
 //   for (iDim = 0; iDim < 3; iDim++)
 //     Center[iDim] += dU[iDim];

 //   config->SetMarkerMotion_Origin(Center, jMarker);
 //   }
    	Center[0] = config->GetRefOriginMoment_X(jMarker);
   	Center[1] = config->GetRefOriginMoment_Y(jMarker);
    	Center[2] = config->GetRefOriginMoment_Z(jMarker);
	
  	cout << "Center= (" ;
    	for (iDim = 0; iDim < 3; iDim++){
	     	cout << Center[iDim] << " " ;
	}
        cout << ")" << endl;

   
      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Index and coordinates of the current point ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();
        //
	if (harmonic_balance){
        for (iDim = 0; iDim < nDim; iDim++){
           su2double val_coord = nodes->GetMesh_Coord(iPoint,iDim);
           geometry->nodes->SetCoord(iPoint, iDim, val_coord);
	}
	}
        Coord  = geometry->nodes->GetCoord(iPoint);
        //GridVel = geometry->nodes->GetGridVel(iPoint);

        /*--- Calculate non-dim. position from rotation center ---*/

	newGridVel[0] = xDot[0];
	newGridVel[1] = xDot[1];
     	if (nDim == 3) newGridVel[2] = xDot[2];

        for (iDim = 0; iDim < nDim; iDim++){
         r[iDim] = (Coord[iDim]-Center[iDim])/Lref;
	}
        /*--- Compute transformed point coordinates ---*/
 
     	Rotate(rotMatrix, Center, r, rotCoord);
     
    	/*--- Cross Product of angular velocity and distance from center.
    	 Note that we have assumed the grid velocities have been set to
         an initial value in the plunging routine. ---*/       

        /*--- Calculate delta change in the x, y, & z directions ---*/
        for (iDim = 0; iDim < nDim; iDim++){
          VarCoord[iDim] = (rotCoord[iDim]-Coord[iDim])/Lref;
          VelCoord[iDim] = (rotCoord[iDim]-Center[iDim])/Lref;
	}
	
	newGridVel[0] +=  - alphaDot[2]*VelCoord[1];
        newGridVel[1] +=  alphaDot[2]*VelCoord[0] ;
        if (nDim == 3) newGridVel[2] += 0;

	/*--- Set node displacement for volume deformation ---*/	

        for (iDim = 0; iDim < nDim; iDim++){

	  if (harmonic_balance) VarCoordAbs[iDim] = dU[iDim] + VarCoord[iDim]; 
//	  else VarCoordAbs[iDim] = dU[iDim] + VarCoord[iDim];
	  else VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + dU[iDim] + VarCoord[iDim];
        }

	//cout << "BND Velo_y = " << newGridVel[1] << endl;

       	//geometry->vertex[iMarker][iVertex]->SetVarCoord(VarCoordAbs);

        nodes->SetBound_Disp(iPoint, VarCoordAbs);

        //if (harmonic_balance) 
//	nodes->SetBound_Vel(iPoint, newGridVel);
      }
      
      for (iDim = 0; iDim < 3; iDim++){
       Center[iDim] += dU[iDim];
       //Center[iDim] += 1.0;
      }

      cout << "Center OUT= (" ;
    	for (iDim = 0; iDim < 3; iDim++){
	     	cout << Center[iDim] << " " ;
	}
        cout << ")" << endl;


///        config->SetMarkerMotion_Origin(Center, jMarker);

        config->SetRefOriginMoment_X_HB(iter, Center[0]);
        config->SetRefOriginMoment_Y_HB(iter, Center[1]);
        config->SetRefOriginMoment_Z_HB(iter, Center[2]); 

	cout << "center set" << endl;
    }
  }
  /*--- For pitching we don't update the motion origin and moment reference origin. ---*/

}

void CMeshSolver::SetStructuralModes(CGeometry *geometry, CConfig *config) {

  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  unsigned short iDim;
  su2double phi_x, phi_y, phi_z, phi_as, phi_ss;
  unsigned long nps = config->GetNumber_STR_Nodes();
  unsigned short STRmodes = config->GetNumber_Modes();
  unsigned short method = config->Get_RBF_method();
  unsigned long dofs = config->Get_STR_Dofs(); // STR is a surface here
  unsigned long kk;

  static int mpi_size = SU2_MPI::GetSize();
  static SU2_MPI::Status mpi_status;

  string f1;
  string filename = config->Get_STR_name(), line;
  string fullfile_mesh = filename + ".vertices";
  string fullfile_mode = filename + ".mode";
  string extension;
  stringstream ss;
	      
  ifstream modefile;
  ifstream meshfile;

  vector<su2double> xa(3,0.0), xs1(3,0.0), xs2(3,0.0);
  su2double dummy1, dummy2, dummy3;
  vector<su2double> PHI_X_str(nps,0.0);
  vector<su2double> PHI_Y_str(nps,0.0);
  vector<su2double> PHI_Z_str(nps,0.0);

  vector<su2double> sol_vec_x(nps+dofs+1,0.0);
  vector<su2double> sol_vec_y(nps+dofs+1,0.0);
  vector<su2double> sol_vec_z(nps+dofs+1,0.0);

  vector<su2double> X_str(nps,0.0);
  vector<su2double> Y_str(nps,0.0);
  vector<su2double> Z_str(nps,0.0);
 
  vector<vector<su2double>> Css(1+dofs+nps,vector<su2double>(1+dofs+nps+1,0.0));

  if (rank == MASTER_NODE) cout << "Setting Structural Modes on Aerodynamic Surface" << endl;
  if (rank == MASTER_NODE) cout << "Mesh filename : " << fullfile_mesh << endl;
     /*--- Store displacement of each node on the plunging surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to plunge ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

    for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if (rank == MASTER_NODE) cout << "Marker : " <<  Marker_Tag << endl;

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }
 
      // READ STR MESH NODES
      //SU2_OMP_MASTER
      if (rank == MASTER_NODE){

	      meshfile.open(fullfile_mesh);

      kk = 0;
      cout << "Reading structural (Surface) nodes." << endl;
      if (meshfile.is_open()) {	
	      
	getline(meshfile,line);

	while ( kk < nps ){
	
	meshfile >> X_str[kk] >> Y_str[kk] >> Z_str[kk];

	kk++;

	}

	meshfile.close();
	if (rank == MASTER_NODE) cout << "Reached End-of-file for Structural Mesh." <<  endl;
	
      }	
      else cout << "Unable to open Mesh file..." << endl; 

#ifdef HAVE_MPI
      for (int destination=1;destination<size;destination++){
	      SU2_MPI::Send(&X_str[0], nps, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
	      SU2_MPI::Send(&Y_str[0], nps, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
	      SU2_MPI::Send(&Z_str[0], nps, MPI_DOUBLE, destination, 0, SU2_MPI::GetComm());
      }
#endif

      // RBF MATRIX
      for (unsigned long ii=0;ii<nps;ii++){

	      Css[0][dofs+1+ii] = 1.0;
	      Css[1][dofs+1+ii] = X_str[ii];
	      Css[2][dofs+1+ii] = Y_str[ii];
	      if (dofs==3) Css[3][dofs+1+ii] = Z_str[ii];

	      Css[dofs+1+ii][0] = 1.0;
	      Css[dofs+1+ii][1] = X_str[ii];
	      Css[dofs+1+ii][2] = Y_str[ii];
	      if (dofs==3) Css[dofs+1+ii][3] = Z_str[ii];
      }

      for (unsigned long ii=0;ii<nps;ii++){
      for (unsigned long jj=0;jj<nps;jj++){
	      
	      xs1[0] = X_str[ii] ; xs1[1] = Y_str[ii] ; xs1[2] = Z_str[ii];
	      xs2[0] = X_str[jj] ; xs2[1] = Y_str[jj] ; xs2[2] = Z_str[jj];

	      phi_ss = RBF_Basis_Function(xs1,xs2,method);

	      Css[dofs+1+ii][dofs+1+jj] = phi_ss;      

      }
      }

      Css[0][nps+dofs+1] = 0.0;
      Css[1][nps+dofs+1] = 0.0;
      Css[2][nps+dofs+1] = 0.0;	      
      if (dofs==3) Css[3][nps+dofs+1] = 0.0;
      }
      else {

#if HAVE_MPI
	      SU2_MPI::Recv(&X_str[0], nps, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
	      SU2_MPI::Recv(&Y_str[0], nps, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
	      SU2_MPI::Recv(&Z_str[0], nps, MPI_DOUBLE, 0, 0, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
#endif

      }
	      
      for (unsigned short mode=0;mode<STRmodes;mode++){
              
      // READ STR MODES
      if (rank == MASTER_NODE) {  

	      ss << mode+1;
	      extension = to_string(mode+1);
	      modefile.open(fullfile_mode+extension);

      kk = 0;
      cout << "Reading Structural Mode " << mode+1 << " from file : " << fullfile_mode+extension << endl; 
      if (modefile.is_open()) {

	getline(modefile,line);

	while (kk < nps) {
	
	modefile >> dummy1 >> dummy2 >> dummy3;

	PHI_X_str[kk] = dummy1;
	PHI_Y_str[kk] = dummy2;
	PHI_Z_str[kk] = dummy3;

	kk++;
	}
	
	modefile.close();
	if (rank == MASTER_NODE) cout << "End-of-file for Mode " <<  mode+1 << endl;
	
      }	
      else cout << "Unable to read Mode... " << endl; 

      if (rank == MASTER_NODE) {
        cout << " Storing Mode " << mode+1 <<  " displacement, for marker : ";
        cout << Marker_Tag << "." << endl; 
      }

      /*--- RBF SYSTEM. ---*/
      for (unsigned long ii=0;ii<nps;ii++) Css[dofs+1+ii][dofs+1+nps] = PHI_X_str[ii]; 
      for (unsigned long ii=0;ii<nps+dofs+1;ii++) sol_vec_x[ii] = 0.0;
 
      Gauss_Elimination(Css, sol_vec_x);
 
      for (unsigned long ii=0;ii<nps;ii++) Css[dofs+1+ii][dofs+1+nps] = PHI_Y_str[ii]; 
      for (unsigned long ii=0;ii<nps+dofs+1;ii++) sol_vec_y[ii] = 0.0;
 
      Gauss_Elimination(Css, sol_vec_y);
 
      for (unsigned long ii=0;ii<nps;ii++) Css[dofs+1+ii][dofs+1+nps] = PHI_Z_str[ii]; 
      for (unsigned long ii=0;ii<nps+dofs+1;ii++) sol_vec_z[ii] = 0.0;
 
      Gauss_Elimination(Css, sol_vec_z);

      cout << "will send solutions " << endl;
       
#if HAVE_MPI
      for (int destination=1;destination<size;destination++) {
	      cout << "sending to proc " << destination << endl;
	      SU2_MPI::Send(&sol_vec_x[0], nps+dofs+1, MPI_DOUBLE, destination, 1, SU2_MPI::GetComm());
	      SU2_MPI::Send(&sol_vec_y[0], nps+dofs+1, MPI_DOUBLE, destination, 2, SU2_MPI::GetComm());
	      SU2_MPI::Send(&sol_vec_z[0], nps+dofs+1, MPI_DOUBLE, destination, 3, SU2_MPI::GetComm());
      }
#endif

      }
      else {
#if HAVE_MPI
       SU2_MPI::Recv(&sol_vec_x[0], nps+dofs+1, MPI_DOUBLE, 0, 1, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
       SU2_MPI::Recv(&sol_vec_y[0], nps+dofs+1, MPI_DOUBLE, 0, 2, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
       SU2_MPI::Recv(&sol_vec_z[0], nps+dofs+1, MPI_DOUBLE, 0, 3, SU2_MPI::GetComm(), MPI_STATUS_IGNORE);
#endif
      }

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

	phi_x = 0.0;
	phi_y = 0.0;
	phi_z = 0.0;

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

	for (iDim = 0; iDim < nDim; iDim++) {
		
		xa[iDim] = geometry->nodes->GetCoord(iPoint,iDim);

		nodes->SetBound_Disp(iPoint, iDim, 0.0);
	}

	phi_x = sol_vec_x[0] + xa[0]*sol_vec_x[1] + xa[1]*sol_vec_x[2];  
      	if (dofs==3) phi_x += xa[2]*sol_vec_x[3];
	
	phi_y = sol_vec_y[0] + xa[0]*sol_vec_y[1] + xa[1]*sol_vec_y[2];  
      	if (dofs==3) phi_y += xa[2]*sol_vec_y[3];

	phi_z = sol_vec_z[0] + xa[0]*sol_vec_z[1] + xa[1]*sol_vec_z[2];  
      	if (dofs==3) phi_z += xa[2]*sol_vec_z[3];

	for (unsigned long ii=0;ii<nps;ii++){

		xs1[0] = X_str[ii] ; xs1[1] = Y_str[ii] ; xs1[2] = Z_str[ii];

		phi_as = RBF_Basis_Function(xa,xs1,method);
 
		phi_x += (phi_as * sol_vec_x[ii+dofs+1]);	
		phi_y += (phi_as * sol_vec_y[ii+dofs+1]);	
		phi_z += (phi_as * sol_vec_z[ii+dofs+1]);	

	}
	
	nodes->SetBound_Mode_X(iPoint, mode, phi_x);
	nodes->SetBound_Mode_Y(iPoint, mode, phi_y);
	nodes->SetBound_Mode_Z(iPoint, mode, phi_z);

      }

      }


    }
  }
 
}

su2double CMeshSolver::RBF_Basis_Function(vector<su2double> x1, vector<su2double> x2, unsigned short method) {

	su2double phi=0.0, dist;

	su2double alpha = 0.01;

	// multi-quadric biharmonic splines
	if (method == 1) {

		phi = (x1[0] - x2[0])*(x1[0] - x2[0]) +
		      (x1[1] - x2[1])*(x1[1] - x2[1]) +
		      (x1[2] - x2[2])*(x1[2] - x2[2]) + alpha*alpha;
	
	}

	// thin plate spline
	if (method == 2) {
	
		dist = (x1[0] - x2[0])*(x1[0] - x2[0]) +
		       (x1[1] - x2[1])*(x1[1] - x2[1]) +
		       (x1[2] - x2[2])*(x1[2] - x2[2]) + alpha*alpha;
		
		phi  = dist * log10(sqrt(dist)); 
	
	}

	// euclid's hat
	if (method == 3) {
	
		su2double r = 0.1;

		dist = (x1[0] - x2[0])*(x1[0] - x2[0]) +
		       (x1[1] - x2[1])*(x1[1] - x2[1]) +
		       (x1[2] - x2[2])*(x1[2] - x2[2]) ;
		
		phi  = PI_NUMBER*( 1.0/12.0 * dist * sqrt(dist) - 
				   r*r*sqrt(dist) + 4.0/3.0*r*r*r ); 
	
	}

	// Hardy's multiquadric
	if (method == 4) {
	
		su2double c = 0.0001;

		dist = (x1[0] - x2[0])*(x1[0] - x2[0]) +
		       (x1[1] - x2[1])*(x1[1] - x2[1]) +
		       (x1[2] - x2[2])*(x1[2] - x2[2]) ;
		
		phi  = sqrt( c*c + dist ); 
	
	}

	return phi;

}

void CMeshSolver::Gauss_Elimination(vector<vector<su2double>> A,vector<su2double>& sol) {
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

void CMeshSolver::Calculate_Surface_Displacement(su2double* gen_disp, CGeometry *geometry, CConfig *config, unsigned long TimeIter) {

  su2double VarCoordAbs[3] = {0.0}, dx[3] = {0.0};
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag;
  unsigned short iDim;
  unsigned short STRmodes = config->GetNumber_Modes();
  unsigned short dofs = 2; // STR is a surface here

  bool harmonic_balance = (config->GetTime_Marching() == TIME_MARCHING::HARMONIC_BALANCE);
  unsigned long kk;

  /*--- Store displacement of each node on the plunging surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to plunge ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

   for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

   /*--- Evaluate reference values for non-dimensionalization.
     For dynamic meshes, use the motion Mach number as a reference value
     for computing the force coefficients. Otherwise, use the freestream values,
     which is the standard convention. ---*/
 
      if (rank == MASTER_NODE) {
      cout << " Calculating Surface Displacement, Z axis component, for marker: ";
      cout << Marker_Tag << "." << endl; 
      cout <<"with Gen. Displacement: ";
      for (int ii=0;ii<STRmodes;ii++) {
      
	      cout << gen_disp[ii] << "| ";

      }
      cout << endl;
      }

      for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[iMarker][iVertex]->GetNode();

	 dx[0] = 0.0;
	 dx[1] = 0.0;
	 dx[2] = 0.0;
	 for (unsigned short mode=0;mode<STRmodes;mode++){

		 dx[0] += (nodes->GetBound_Mode_X(iPoint, mode)*gen_disp[mode]);
                 dx[1] += (nodes->GetBound_Mode_Y(iPoint, mode)*gen_disp[mode]);
		 dx[2] += (nodes->GetBound_Mode_Z(iPoint, mode)*gen_disp[mode]);
     	
	 }

  	 for (iDim = 0; iDim < nDim; iDim++) {
                 if (harmonic_balance) VarCoordAbs[iDim] = dx[iDim];
                 else VarCoordAbs[iDim] = nodes->GetBound_Disp(iPoint, iDim) + dx[iDim];	
      	 }

      	 nodes->SetBound_Disp(iPoint, VarCoordAbs);
 
      }

    }
  }
 
}

void CMeshSolver::Calculate_Generalized_Forces(su2double* &gen_forces, CGeometry *geometry, CSolver *flow_solution, CConfig *config, unsigned long TimeIter) {

  su2double VarCoordAbs[3] = {0.0}, dx[3] = {0.0};
  const su2double DEG2RAD = PI_NUMBER/180.0;
  unsigned short iMarker, jMarker;
  unsigned long iPoint, iVertex;
  string Marker_Tag, Moving_Tag, Monitoring_Tag;
  unsigned short iDim;
  unsigned short STRmodes = config->GetNumber_Modes();
  unsigned short dofs = 2; // STR is a surface here

  su2double cl_tot, cd_tot, cl_proc, cd_proc;
  bool write_check=0;
  unsigned long kk;
  if (TimeIter == 5) write_check =1;

  ofstream Valid_file;

  if (write_check) {
    Valid_file.precision(15);
    Valid_file.open("check_pressure_modes.csv", ios::out);
  }

  unsigned short Boundary, Monitoring, iMarker_Monitoring;
  su2double Pressure = 0.0, factor, RefVel2 = 0.0, RefTemp, RefDensity = 0.0, RefPressure, Mach2Vel,
            Mach_Motion, RefArea;
  const su2double *Normal = nullptr, *Coord = nullptr;
        
  su2double Force[3] = {0.0};
  su2double ForceTotal[3] = {0.0};
 
    RefTemp     = config->GetTemperature_FreeStream();
    RefDensity  = config->GetDensity_FreeStream();
    RefPressure = config->GetPressure_FreeStream();
    RefArea     = config->GetRefArea();

    su2double Alpha = config->GetAoA() * PI_NUMBER / 180.0;

    CVariable* flow_nodes = flow_solution->GetNodes(); 

    Mach2Vel    = sqrt(1.4 * config->GetGas_Constant() * RefTemp);  
    Mach_Motion = config->GetMach_Motion();
      
    RefVel2 = (Mach_Motion * Mach2Vel) * (Mach_Motion * Mach2Vel);
   
    factor = 1.0 / (0.5 * RefDensity * RefArea * RefVel2); 
    //factor = 1.0 / (0.5 * RefDensity * RefVel2);

  /*--- Store displacement of each node on the plunging surface ---*/
  /*--- Loop over markers and find the particular marker(s) (surface) to plunge ---*/

  for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    if (config->GetMarker_All_Moving(iMarker) != YES) continue;

    Marker_Tag = config->GetMarker_All_TagBound(iMarker);

   for (jMarker = 0; jMarker < config->GetnMarker_Moving(); jMarker++) {

      Moving_Tag = config->GetMarker_Moving_TagBound(jMarker);

      if ((Marker_Tag != Moving_Tag) || (config->GetKind_SurfaceMovement(jMarker) != DEFORMING)) {
        continue;
      }

   /*--- Evaluate reference values for non-dimensionalization.
     For dynamic meshes, use the motion Mach number as a reference value
     for computing the force coefficients. Otherwise, use the freestream values,
     which is the standard convention. ---*/

      for (unsigned short mode=0;mode<STRmodes;mode++) gen_forces[mode]=0.0;

      if (rank == MASTER_NODE) {
        cout << " Calculating Gen. Forces, Z axis component, for marker: ";
        cout << Marker_Tag << "." << endl; 
      }

      //for (iVertex = 0; iVertex < geometry->nVertex[iMarker]; iVertex++) {
      for (iVertex = 0; iVertex < geometry->GetnVertex(jMarker); iVertex++) {

        /*--- Set node displacement for volume deformation ---*/

        iPoint = geometry->vertex[jMarker][iVertex]->GetNode();

	Pressure = flow_nodes->GetPressure(iPoint);

        if (geometry->nodes->GetDomain(iPoint)) 
	{
	Normal = geometry->vertex[jMarker][iVertex]->GetNormal();
	Coord = geometry->nodes->GetCoord(iPoint);

          for (iDim = 0; iDim < nDim; iDim++) {
            Force[iDim] = -(Pressure - RefPressure) * Normal[iDim] * factor ;
            //Force[iDim] = -(Pressure - RefPressure) * Normal[iDim] ;

            ForceTotal[iDim] += Force[iDim];
          }

 	  for (unsigned short mode=0;mode<STRmodes;mode++)
		 gen_forces[mode] += (nodes->GetBound_Mode_Z(iPoint, mode)*
				      (cos(Alpha)*Force[2]-sin(Alpha)*Force[0])*RefArea +
				      nodes->GetBound_Mode_X(iPoint, mode)*
				      (cos(Alpha)*Force[0]+sin(Alpha)*Force[2])*RefArea);
	} 
      }

      cl_tot = (cos(Alpha)*ForceTotal[2]-sin(Alpha)*ForceTotal[0]);
      cd_tot = (cos(Alpha)*ForceTotal[0]+sin(Alpha)*ForceTotal[2]);

#ifdef HAVE_MPI

      if (config->GetComm_Level() == COMM_FULL) {

	      auto Allreduce = [](su2double x) {
              su2double tmp = x;
              x = 0.0;
              SU2_MPI::Allreduce(&tmp, &x, 1, MPI_DOUBLE, MPI_SUM, SU2_MPI::GetComm());
              return x;  
              };

	      cl_tot = Allreduce(cl_tot);
	      cd_tot = Allreduce(cd_tot);

	      for (unsigned short mode=0;mode<STRmodes;mode++)
		      gen_forces[mode] = Allreduce(gen_forces[mode]);
      }

#endif

      if (rank == MASTER_NODE) 
	 cout <<  "in gen. forces : CL_tot= " << cl_tot << " | CD_tot= " << cd_tot << endl;

    }
  }
 
}
