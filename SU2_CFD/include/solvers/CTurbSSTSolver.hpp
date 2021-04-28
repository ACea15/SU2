/*!
 * \file CTurbSSTSolver.hpp
 * \brief Headers of the CTurbSSTSolver class
 * \author A. Campos, F. Palacios, T. Economon
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

#pragma once

#include "CTurbSolver.hpp"

/*!
 * \class CTurbSSTSolver
 * \brief Main class for defining the turbulence model solver.
 * \ingroup Turbulence_Model
 * \author A. Campos, F. Palacios, T. Economon
 */
class CTurbSSTSolver final : public CTurbSolver {
private:
  su2double
  constants[10] = {0.0}, /*!< \brief Constants for the model. */
  kine_Inf,              /*!< \brief Free-stream turbulent kinetic energy. */
  omega_Inf;             /*!< \brief Free-stream specific dissipation. */

  su2double *Primitive,    /*!< \brief Auxiliary nPrimVar vector. */
  *Primitive_i,            /*!< \brief Auxiliary nPrimVar vector for storing the primitive at point i. */
  *Primitive_j;            /*!< \brief Auxiliary nPrimVar vector for storing the primitive at point j. */

public:
  /*!
   * \brief Constructor of the class.
   */
  CTurbSSTSolver(void);

  /*!
   * \overload
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  CTurbSSTSolver(CGeometry *geometry, CConfig *config, unsigned short iMesh);

  /*!
   * \brief Destructor of the class.
   */
  ~CTurbSSTSolver(void) override;

  /*!
   * \brief Restart residual and compute gradients.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   * \param[in] iRKStep - Current step of the Runge-Kutta iteration.
   * \param[in] RunTime_EqSystem - System of equations which is going to be solved.
   * \param[in] Output - boolean to determine whether to print output.
   */
  void Preprocessing(CGeometry *geometry,
                     CSolver **solver,
                     CConfig *config,
                     unsigned short iMesh,
                     unsigned short iRKStep,
                     unsigned short RunTime_EqSystem,
                     bool Output) override;

  /*!
   * \brief Computes the eddy viscosity.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Postprocessing(CGeometry *geometry,
                      CSolver **solver,
                      CConfig *config,
                      unsigned short iMesh) override;
  
  void SetPrimitive_Variables(CSolver **solver) final;
  
  void SetEddyViscosity(CGeometry *geometry, CSolver **solver);
  
  /*!
   * \brief Compute the gradient of the primitive variables using Green-Gauss method,
   *        and stores the result in the <i>Gradient_Primitive</i> variable.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] reconstruction - indicator that the gradient being computed is for upwind reconstruction.
   */
  void SetPrimitive_Gradient_GG(CGeometry *geometry,
                                CConfig *config,
                                bool reconstruction = false) final;

  /*!
   * \brief Compute the gradient of the primitive variables using a Least-Squares method,
   *        and stores the result in the <i>Gradient_Primitive</i> variable.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] reconstruction - indicator that the gradient being computed is for upwind reconstruction.
   */
  void SetPrimitive_Gradient_LS(CGeometry *geometry,
                                CConfig *config,
                                bool reconstruction = false) final;

  /*!
   * \brief Compute the limiter of the primitive variables.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   */
  void SetPrimitive_Limiter(CGeometry *geometry, CConfig *config) final;

  /*!
   * \brief Source term computation.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] numerics_container - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Source_Residual(CGeometry *geometry,
                       CSolver **solver,
                       CNumerics **numerics_container,
                       CConfig *config,
                       unsigned short iMesh) override;

  /*!
  * \brief Jacobians with respect to the gradients in the cross diffusion term.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] solver - Container vector with all the solutions.
  * \param[in] config - Definition of the particular problem.
  */
  void CrossDiffusionJacobian(CSolver         **solver,
                              const CGeometry *geometry,
                              const CConfig   *config,
                              const unsigned long iPoint);

  /*!
   * \brief Source term computation.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   */
  void Source_Template(CGeometry *geometry,
                       CSolver **solver,
                       CNumerics *numerics,
                       CConfig *config,
                       unsigned short iMesh) override;

  /*!
   * \brief Impose the Navier-Stokes wall boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_HeatFlux_Wall(CGeometry *geometry,
                        CSolver **solver,
                        CNumerics *conv_numerics,
                        CNumerics *visc_numerics,
                        CConfig *config,
                        unsigned short val_marker) override;
  /*!
   * \brief Impose the Navier-Stokes wall boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Isothermal_Wall(CGeometry *geometry,
                          CSolver **solver,
                          CNumerics *conv_numerics,
                          CNumerics *visc_numerics,
                          CConfig *config,
                          unsigned short val_marker) override;

  /*!
   * \brief Impose the Far Field boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Far_Field(CGeometry *geometry,
                    CSolver **solver,
                    CNumerics *conv_numerics,
                    CNumerics *visc_numerics,
                    CConfig *config,
                    unsigned short val_marker) override;

  /*!
   * \brief Impose the inlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Inlet(CGeometry *geometry,
                CSolver **solver,
                CNumerics *conv_numerics,
                CNumerics *visc_numerics,
                CConfig *config,
                unsigned short val_marker) override;

  /*!
   * \brief Impose the inlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Inlet_Turbo(CGeometry *geometry,
                      CSolver **solver,
                      CNumerics *conv_numerics,
                      CNumerics *visc_numerics,
                      CConfig *config,
                      unsigned short val_marker) override;

  /*!
   * \brief Impose the inlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Inlet_MixingPlane(CGeometry *geometry,
                            CSolver **solver,
                            CNumerics *conv_numerics,
                            CNumerics *visc_numerics,
                            CConfig *config,
                            unsigned short val_marker) override;
  /*!
   * \brief Impose the outlet boundary condition.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] conv_numerics - Description of the numerical method.
   * \param[in] visc_numerics - Description of the numerical method.
   * \param[in] config - Definition of the particular problem.
   * \param[in] val_marker - Surface marker where the boundary condition is applied.
   */
  void BC_Outlet(CGeometry *geometry,
                 CSolver **solver,
                 CNumerics *conv_numerics,
                 CNumerics *visc_numerics,
                 CConfig *config,
                 unsigned short val_marker) override;
 /*!
  * \brief Impose the interface state across sliding meshes.
  * \param[in] geometry - Geometrical definition of the problem.
  * \param[in] solver - Container vector with all the solutions.
  * \param[in] conv_numerics - Description of the numerical method.
  * \param[in] visc_numerics - Description of the numerical method.
  * \param[in] config - Definition of the particular problem.
  */
  void BC_Fluid_Interface(CGeometry *geometry,
                          CSolver **solver,
                          CNumerics *conv_numerics,
                          CNumerics *visc_numerics,
                          CConfig *config) override;

  /*!
   * \brief Get the constants for the SST model.
   * \return A pointer to an array containing a set of constants
   */
  inline const su2double* GetConstants() const override { return constants; }

  /*!
   * \brief Set the solution using the Freestream values.
   * \param[in] config - Definition of the particular problem.
   */
  inline void SetFreeStream_Solution(CConfig *config) override {
    for (unsigned long iPoint = 0; iPoint < nPoint; iPoint++){
      nodes->SetSolution(iPoint, 0, kine_Inf);
      nodes->SetSolution(iPoint, 1, omega_Inf);
    }
  }


  /*!
   * \brief Store of a set of provided inlet profile values at a vertex.
   * \param[in] val_inlet - vector containing the inlet values for the current vertex.
   * \param[in] iMarker - Surface marker where the coefficient is computed.
   * \param[in] iVertex - Vertex of the marker <i>iMarker</i> where the inlet is being set.
   */
  void SetInletAtVertex(su2double *val_inlet,
                        unsigned short iMarker,
                        unsigned long iVertex) override;

  /*!
   * \brief Get the set of value imposed at an inlet.
   * \param[in] val_inlet - vector returning the inlet values for the current vertex.
   * \param[in] val_inlet_point - Node index where the inlet is being set.
   * \param[in] val_kind_marker - Enumerated type for the particular inlet type.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param config - Definition of the particular problem.
   * \return Value of the face area at the vertex.
   */
  su2double GetInletAtVertex(su2double *val_inlet,
                             unsigned long val_inlet_point,
                             unsigned short val_kind_marker,
                             string val_marker,
                             CGeometry *geometry,
                             CConfig *config) const override;
  /*!
   * \brief Set a uniform inlet profile
   *
   * The values at the inlet are set to match the values specified for
   * inlets in the configuration file.
   *
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMarker - Surface marker where the coefficient is computed.
   */
  void SetUniformInlet(CConfig* config, unsigned short iMarker) override;

  /*!
   * \brief Get the value of the turbulent kinetic energy.
   * \return Value of the turbulent kinetic energy.
   */
  inline su2double GetTke_Inf(void) const override { return kine_Inf; }

  /*!
   * \brief Get the value of the turbulent frequency.
   * \return Value of the turbulent frequency.
   */
  inline su2double GetOmega_Inf(void) const override { return omega_Inf; }

  /*!
   * \brief Compute the time step for solving the Euler equations.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iMesh - Index of the mesh in multigrid computations.
   * \param[in] Iteration - Value of the current iteration.
   */
  // void SetTime_Step(CGeometry *geometry,
  //                   CSolver **solver,
  //                   CConfig *config,
  //                   unsigned short iMesh,
  //                   unsigned long Iteration) final;
  
  /*!
   * \brief Computes the wall shear stress (Tau_Wall) on the surface using a wall function.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   */
  void ComputeNicholsWallFunction(CGeometry *geometry,
                                  CSolver** solver,
                                  CConfig* config) override;

  /*!
   * \brief Computes the wall shear stress (Tau_Wall) on the surface using a wall function.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] solver - Container vector with all the solutions.
   * \param[in] config - Definition of the particular problem.
   */
  void ComputeKnoppWallFunction(CGeometry *geometry,
                                CSolver** solver,
                                CConfig* config) override;

  /*!
   * \brief Compute the turbulent terms of the goal-oriented metric.
   * \param[in] solver - Physical definition of the problem.
   * \param[in] geometry - Geometrical definition of the problem.
   * \param[in] config - Definition of the particular problem.
   * \param[in] iPoint - Index of current node.
   * \param[in] weights - Weights of each Hessian in the metric.
   */
  void TurbulentMetric(CSolver                    **solver,
                       CGeometry                  *geometry,
                       CConfig                    *config,
                       unsigned long              iPoint,
                       vector<vector<su2double> > &weights);

};
