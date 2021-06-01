/*!
 * \file CDeformationDriver.hpp
 * \brief Main subroutines for driving the mesh deformation.
 * \author T. Economon, H. Kline, R. Sanchez
 * \version 7.1.1 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2021, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser/ General Public
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


#include "../../include/drivers/CDeformationDriver.hpp"

#include "../../../Common/include/geometry/CPhysicalGeometry.hpp"
#include "../../../SU2_CFD/include/output/CMeshOutput.hpp"

using namespace std;

CDeformationDriver::CDeformationDriver(char* confFile, SU2_Comm MPICommunicator) {

  /*--- Initialize Medipack (must also be here so it is initialized from python) ---*/
  #ifdef HAVE_MPI
    #if defined(CODI_REVERSE_TYPE) || defined(CODI_FORWARD_TYPE)
      SU2_MPI::Init_AMPI();
    #endif
  #endif

  SU2_MPI::SetComm(MPICommunicator);

  rank = SU2_MPI::GetRank();
  size = SU2_MPI::GetSize();

  /*--- Copy the config filename ---*/
  strcpy(config_file_name, confFile);

  /*--- Initialize the configuration of the driver ---*/
  driver_config = nullptr;
  driver_config = new CConfig(config_file_name, SU2_COMPONENT::SU2_DEF);

  nZone = driver_config->GetnZone();

  /*--- Initialize containers --- */

  SetContainers_Null();

  /*--- Preprocessing of the config files. ---*/

  Input_Preprocessing();

  /*--- Set up a timer for performance benchmarking ---*/

  StartTime = SU2_MPI::Wtime();

  /*--- Preprocessing of the geometry for all zones. ---*/

  Geometrical_Preprocessing();

  /*--- Preprocessing of the output for all zones. ---*/

  Output_Preprocessing();

  /*--- Preprocessing time is reported now, but not included in the next compute portion. ---*/

  StopTime = SU2_MPI::Wtime();

  /*--- Compute/print the total time for performance benchmarking. ---*/

  UsedTime = StopTime-StartTime;
  UsedTimePreproc    = UsedTime;
  UsedTimeCompute    = 0.0;

}

CDeformationDriver::~CDeformationDriver(void) {

}

void CDeformationDriver::SetContainers_Null() {

    /*--- Create pointers to all of the classes that may be used throughout
        the SU2_DEF code. In general, the pointers are instantiated down a
        hierarchy over all zones as described in the comments below. ---*/
    config_container   = new CConfig*[nZone];
    geometry_container = new CGeometry*[nZone];
    surface_movement   = new CSurfaceMovement*[nZone];
    grid_movement      = new CVolumetricMovement*[nZone];
    output_container   = new COutput*[nZone];

    for (iZone = 0; iZone < nZone; iZone++) {
        config_container[iZone]       = nullptr;
        geometry_container[iZone]     = nullptr;
        surface_movement[iZone]       = nullptr;
        grid_movement[iZone]          = nullptr;
        output_container[iZone]       = nullptr;
    }
}

void CDeformationDriver::Input_Preprocessing() {

    /*--- Initialize a char to store the zone filename ---*/
    char zone_file_name[MAX_STRING_SIZE];

    /*--- Loop over all zones to initialize the various classes. In most
       cases, nZone is equal to one. This represents the solution of a partial
       differential equation on a single block, unstructured mesh. ---*/

    for (iZone = 0; iZone < nZone; iZone++) {

    /*--- Definition of the configuration option class for all zones. In this
          constructor, the input configuration file is parsed and all options are
          read and stored. ---*/

        if (driver_config->GetnConfigFiles() > 0){
            strcpy(zone_file_name, driver_config->GetConfigFilename(iZone).c_str());
            config_container[iZone] = new CConfig(driver_config, zone_file_name, SU2_COMPONENT::SU2_DEF, iZone, nZone, true);
        } else {
            config_container[iZone] = new CConfig(driver_config, config_file_name, SU2_COMPONENT::SU2_DEF, iZone, nZone, true);
        }

        config_container[iZone]->SetMPICommunicator(SU2_MPI::GetComm());
    }

      /*--- Set the multizone part of the problem. ---*/
    if (driver_config->GetMultizone_Problem()){
        for (iZone = 0; iZone < nZone; iZone++) {
            /*--- Set the interface markers for multizone ---*/
            config_container[iZone]->SetMultizone(driver_config, config_container);
        }
    }
}

void CDeformationDriver::Geometrical_Preprocessing() {

    for (iZone = 0; iZone < nZone; iZone++) {

    /*--- Definition of the geometry class to store the primal grid in the partitioning process. ---*/

    CGeometry *geometry_aux = nullptr;

    /*--- All ranks process the grid and call ParMETIS for partitioning ---*/

    geometry_aux = new CPhysicalGeometry(config_container[iZone], iZone, nZone);

    /*--- Color the initial grid and set the send-receive domains (ParMETIS) ---*/

    geometry_aux->SetColorGrid_Parallel(config_container[iZone]);

    /*--- Build the grid data structures using the ParMETIS coloring. ---*/

    geometry_container[iZone] = new CPhysicalGeometry(geometry_aux, config_container[iZone]);

    /*--- Deallocate the memory of geometry_aux ---*/

    delete geometry_aux;

    /*--- Add the Send/Receive boundaries ---*/

    geometry_container[iZone]->SetSendReceive(config_container[iZone]);

    /*--- Add the Send/Receive boundaries ---*/

    geometry_container[iZone]->SetBoundaries(config_container[iZone]);

    /*--- Computational grid preprocesing ---*/

    if (rank == MASTER_NODE) cout << endl << "----------------------- Preprocessing computations ----------------------" << endl;

    /*--- Compute elements surrounding points, points surrounding points ---*/

    if (rank == MASTER_NODE) cout << "Setting local point connectivity." <<endl;
    geometry_container[iZone]->SetPoint_Connectivity();

    /*--- Check the orientation before computing geometrical quantities ---*/

    geometry_container[iZone]->SetBoundVolume();
    if (config_container[iZone]->GetReorientElements()) {
    if (rank == MASTER_NODE) cout << "Checking the numerical grid orientation of the interior elements." <<endl;
        geometry_container[iZone]->Check_IntElem_Orientation(config_container[iZone]);
        geometry_container[iZone]->Check_BoundElem_Orientation(config_container[iZone]);
    }

    /*--- Create the edge structure ---*/

    if (rank == MASTER_NODE) cout << "Identify edges and vertices." <<endl;
    geometry_container[iZone]->SetEdges();
    geometry_container[iZone]->SetVertex(config_container[iZone]);

    if (config_container[iZone]->GetDesign_Variable(0) != NO_DEFORMATION) {

        /*--- Create the dual control volume structures ---*/

        if (rank == MASTER_NODE) cout << "Setting the bound control volume structure." << endl;
        geometry_container[iZone]->SetControlVolume(config_container[iZone], ALLOCATE);
        geometry_container[iZone]->SetBoundControlVolume(config_container[iZone], ALLOCATE);
    }

    /*--- Create the point-to-point MPI communication structures. ---*/

    geometry_container[iZone]->PreprocessP2PComms(geometry_container[iZone], config_container[iZone]);

    }
}

void CDeformationDriver::Output_Preprocessing() {

      for (iZone = 0; iZone < nZone; iZone++) {

        /*--- Allocate the mesh output ---*/

        output_container[iZone] = new CMeshOutput(config_container[iZone], geometry_container[iZone]->GetnDim());

        /*--- Preprocess the volume output ---*/

        output_container[iZone]->PreprocessVolumeOutput(config_container[iZone]);

        /*--- Preprocess history --- */

        output_container[iZone]->PreprocessHistoryOutput(config_container[iZone], false);

    }
}

void CDeformationDriver::Run() {

    /* --- Start measuring computation time ---*/

    StartTime = SU2_MPI::Wtime();

    /*--- Surface grid deformation using design variables ---*/

    for (iZone = 0; iZone < nZone; iZone++){

      if (config_container[iZone]->GetDesign_Variable(0) != NO_DEFORMATION) {

        /*--- Definition of the Class for grid movement ---*/
        grid_movement[iZone] = new CVolumetricMovement(geometry_container[iZone], config_container[iZone]);

        /*--- Save original coordinates to be reused in convexity checking procedure ---*/
        auto OriginalCoordinates = geometry_container[iZone]->nodes->GetCoord();

        /*--- First check for volumetric grid deformation/transformations ---*/

        if (config_container[iZone]->GetDesign_Variable(0) == SCALE_GRID) {

          if (rank == MASTER_NODE)
            cout << endl << "--------------------- Volumetric grid scaling (ZONE " << iZone <<") ------------------" << endl;
          grid_movement[iZone]->SetVolume_Scaling(geometry_container[iZone], config_container[iZone], false);

        } else if (config_container[iZone]->GetDesign_Variable(0) == TRANSLATE_GRID) {

          if (rank == MASTER_NODE)
            cout << endl << "------------------- Volumetric grid translation (ZONE " << iZone <<") ----------------" << endl;
          grid_movement[iZone]->SetVolume_Translation(geometry_container[iZone], config_container[iZone], false);

        } else if (config_container[iZone]->GetDesign_Variable(0) == ROTATE_GRID) {

          if (rank == MASTER_NODE)
            cout << endl << "--------------------- Volumetric grid rotation (ZONE " << iZone <<") -----------------" << endl;
          grid_movement[iZone]->SetVolume_Rotation(geometry_container[iZone], config_container[iZone], false);

        } else {

          /*--- If no volume-type deformations are requested, then this is a
           surface-based deformation or FFD set up. ---*/

          if (rank == MASTER_NODE)
            cout << endl << "--------------------- Surface grid deformation (ZONE " << iZone <<") -----------------" << endl;

          /*--- Definition and initialization of the surface deformation class ---*/

          surface_movement[iZone] = new CSurfaceMovement();

          /*--- Copy coordinates to the surface structure ---*/

          surface_movement[iZone]->CopyBoundary(geometry_container[iZone], config_container[iZone]);

          /*--- Surface grid deformation ---*/

          if (rank == MASTER_NODE) cout << "Performing the deformation of the surface grid." << endl;
          auto TotalDeformation = surface_movement[iZone]->SetSurface_Deformation(geometry_container[iZone], config_container[iZone]);

          if (config_container[iZone]->GetDesign_Variable(0) != FFD_SETTING) {

            if (rank == MASTER_NODE)
              cout << endl << "------------------- Volumetric grid deformation (ZONE " << iZone <<") ----------------" << endl;

            if (rank == MASTER_NODE)
              cout << "Performing the deformation of the volumetric grid." << endl;
            grid_movement[iZone]->SetVolume_Deformation(geometry_container[iZone], config_container[iZone], false);

            /*--- Get parameters for convexity check ---*/
            bool ConvexityCheck;
            unsigned short ConvexityCheck_MaxIter, ConvexityCheck_MaxDepth;

            tie(ConvexityCheck, ConvexityCheck_MaxIter, ConvexityCheck_MaxDepth) = config_container[iZone]->GetConvexityCheck();

            /*--- Recursively change deformations if there are nonconvex elements. ---*/

            if (ConvexityCheck && geometry_container[iZone]->GetnNonconvexElements() > 0) {
              if (rank == MASTER_NODE) {
                cout << "Nonconvex elements present after deformation. " << endl;
                cout << "Recursively lowering deformation magnitude." << endl;
              }

              /*--- Load initial deformation values ---*/
              auto InitialDeformation = TotalDeformation;

              unsigned short ConvexityCheckIter, RecursionDepth = 0;
              su2double DeformationFactor = 1.0, DeformationDifference = 1.0;
              for (ConvexityCheckIter = 1; ConvexityCheckIter <= ConvexityCheck_MaxIter; ConvexityCheckIter++) {

                /*--- Recursively change deformation magnitude:
                decrease if there are nonconvex elements, increase otherwise ---*/
                DeformationDifference /= 2.0;

                if (geometry_container[iZone]->GetnNonconvexElements() > 0) {
                  DeformationFactor -= DeformationDifference;
                } else {
                  RecursionDepth += 1;

                  if (RecursionDepth == ConvexityCheck_MaxDepth) {
                    if (rank == MASTER_NODE) {
                      cout << "Maximum recursion depth reached." << endl;
                      cout << "Remaining amount of original deformation: ";
                      cout << DeformationFactor*100.0 << " percent. " << endl;
                    }
                    break;
                  }

                  DeformationFactor += DeformationDifference;
                }

                /*--- Load mesh to start every iteration with an undeformed grid ---*/
                for (auto iPoint = 0ul; iPoint < OriginalCoordinates.rows(); iPoint++) {
                  for (auto iDim = 0ul; iDim < OriginalCoordinates.cols(); iDim++) {
                    geometry_container[iZone]->nodes->SetCoord(iPoint, iDim, OriginalCoordinates(iPoint,iDim));
                  }
                }

                /*--- Set deformation magnitude as percentage of initial deformation ---*/
                for (auto iDV = 0u; iDV < driver_config->GetnDV(); iDV++) {
                  for (auto iDV_Value = 0u; iDV_Value < driver_config->GetnDV_Value(iDV); iDV_Value++) {
                    config_container[iZone]->SetDV_Value(iDV, iDV_Value, InitialDeformation[iDV][iDV_Value]*DeformationFactor);
                  }
                }

                /*--- Surface grid deformation ---*/
                if (rank == MASTER_NODE) cout << "Performing the deformation of the surface grid." << endl;

                TotalDeformation = surface_movement[iZone]->SetSurface_Deformation(geometry_container[iZone], config_container[iZone]);

                if (rank == MASTER_NODE)
                  cout << endl << "------------------- Volumetric grid deformation (ZONE " << iZone <<") ----------------" << endl;

                if (rank == MASTER_NODE)
                  cout << "Performing the deformation of the volumetric grid." << endl;
                grid_movement[iZone]->SetVolume_Deformation(geometry_container[iZone], config_container[iZone], false);

                if (rank == MASTER_NODE) {
                  cout << "Number of nonconvex elements for iteration " << ConvexityCheckIter << ": ";
                  cout << geometry_container[iZone]->GetnNonconvexElements() << endl;
                  cout << "Remaining amount of original deformation: ";
                  cout << DeformationFactor*100.0 << " percent. " << endl;
                }

              }

            }

          }

        }

      }

    }

    /*--- Computational grid preprocesing ---*/

    if (rank == MASTER_NODE) cout << endl << "----------------------- Write deformed grid files -----------------------" << endl;

    /*--- Output deformed grid for visualization, if requested (surface and volumetric), in parallel
     requires to move all the data to the master node---*/

    for (iZone = 0; iZone < nZone; iZone++){

      /*--- Compute Mesh Quality if requested. Necessary geometry preprocessing re-done beforehand. ---*/

      if (config_container[iZone]->GetWrt_MeshQuality() && !driver_config->GetStructuralProblem()) {

        if (rank == MASTER_NODE) cout << "Recompute geometry properties necessary to evaluate mesh quality statistics.\n";

        geometry_container[iZone]->SetPoint_Connectivity();
        geometry_container[iZone]->SetBoundVolume();
        geometry_container[iZone]->SetEdges();
        geometry_container[iZone]->SetVertex(config_container[iZone]);
        geometry_container[iZone]->SetControlVolume(config_container[iZone], ALLOCATE);
        geometry_container[iZone]->SetBoundControlVolume(config_container[iZone], ALLOCATE);

        if (rank == MASTER_NODE) cout << "Computing mesh quality statistics for the dual control volumes.\n";
        geometry_container[iZone]->ComputeMeshQualityStatistics(config_container[iZone]);
      }// Mesh Quality Output

      /*--- Load the data --- */

      output_container[iZone]->Load_Data(geometry_container[iZone], config_container[iZone], nullptr);

      output_container[iZone]->WriteToFile(config_container[iZone], geometry_container[iZone], MESH, driver_config->GetMesh_Out_FileName());

      /*--- Set the file names for the visualization files ---*/

      output_container[iZone]->SetVolume_Filename("volume_deformed");
      output_container[iZone]->SetSurface_Filename("surface_deformed");

      for (unsigned short iFile = 0; iFile < config_container[iZone]->GetnVolumeOutputFiles(); iFile++){
        auto FileFormat = config_container[iZone]->GetVolumeOutputFiles();
        if (FileFormat[iFile] != RESTART_ASCII && FileFormat[iFile] != RESTART_BINARY)
          output_container[iZone]->WriteToFile(config_container[iZone], geometry_container[iZone], FileFormat[iFile]);
      }
    }

    if ((config_container[ZONE_0]->GetDesign_Variable(0) != NO_DEFORMATION) &&
        (config_container[ZONE_0]->GetDesign_Variable(0) != SCALE_GRID)     &&
        (config_container[ZONE_0]->GetDesign_Variable(0) != TRANSLATE_GRID) &&
        (config_container[ZONE_0]->GetDesign_Variable(0) != ROTATE_GRID)) {

      /*--- Write the the free-form deformation boxes after deformation. ---*/

      if (rank == MASTER_NODE) cout << "Adding any FFD information to the SU2 file." << endl;

      surface_movement[ZONE_0]->WriteFFDInfo(surface_movement, geometry_container, config_container);

    }

    /*--- Synchronization point after a single solver iteration. Compute the
      wall clock time required. ---*/

    StopTime = SU2_MPI::Wtime();

    UsedTimeCompute = StopTime-StartTime;
    if (rank == MASTER_NODE) {
      cout << "\nCompleted in " << fixed << UsedTimeCompute << " seconds on "<< size;
      if (size == 1) cout << " core." << endl; else cout << " cores." << endl;
    }
}

void CDeformationDriver::Postprocessing() {
    if (rank == MASTER_NODE)
      cout << endl <<"------------------------- Solver Postprocessing -------------------------" << endl;

    delete driver_config;
    driver_config = nullptr;

    if (geometry_container != nullptr) {
      for (iZone = 0; iZone < nZone; iZone++) {
        if (geometry_container[iZone] != nullptr) {
          delete geometry_container[iZone];
        }
      }
      delete [] geometry_container;
    }
    if (rank == MASTER_NODE) cout << "Deleted CGeometry container." << endl;

    if (surface_movement != nullptr) {
      for (iZone = 0; iZone < nZone; iZone++) {
        if (surface_movement[iZone] != nullptr) {
          delete surface_movement[iZone];
        }
      }
      delete [] surface_movement;
    }
    if (rank == MASTER_NODE) cout << "Deleted CSurfaceMovement class." << endl;

    if (grid_movement != nullptr) {
      for (iZone = 0; iZone < nZone; iZone++) {
        if (grid_movement[iZone] != nullptr) {
          delete grid_movement[iZone];
        }
      }
      delete [] grid_movement;
    }
    if (rank == MASTER_NODE) cout << "Deleted CVolumetricMovement class." << endl;

    if (config_container != nullptr) {
      for (iZone = 0; iZone < nZone; iZone++) {
        if (config_container[iZone] != nullptr) {
          delete config_container[iZone];
        }
      }
      delete [] config_container;
    }
    if (output_container != nullptr) {
      for (iZone = 0; iZone < nZone; iZone++) {
        if (output_container[iZone] != nullptr) {
          delete output_container[iZone];
        }
      }
      delete [] output_container;
    }
    if (rank == MASTER_NODE) cout << "Deleted CConfig container." << endl;

    if (rank == MASTER_NODE) cout << "Deleted COutput class." << endl;

    /*--- Exit the solver cleanly ---*/

    if (rank == MASTER_NODE)
      cout << endl << "------------------------- Exit Success (SU2_DEF) ------------------------" << endl << endl;
}
