/*!
 * \file CPrimalGridFEM.cpp
 * \brief Class for defining the primal grid element of the FEM solver
 * \author E. van der Weide
 * \version 7.0.7 "Blackbird"
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

#include "../../../include/geometry/primal_grid/CPrimalGridFEM.hpp"
#include "../../../include/fem/CFEMStandardElementBase.hpp"

CPrimalGridFEM::~CPrimalGridFEM() {

  delete[] ElementOwnsFace;
  delete[] PeriodIndexNeighbors;
  delete[] JacobianFaceIsConstant;
}

CPrimalGridFEM::CPrimalGridFEM(const unsigned long *dataElem)
  : CPrimalGrid() {

  /*--- Store the meta data for this element. ---*/
  VTK_Type     = (unsigned short) dataElem[0];
  nPolyGrid    = (unsigned short) dataElem[1];
  nPolySol     = (unsigned short) dataElem[2];
  nDOFsGrid    = (unsigned short) dataElem[3];
  nDOFsSol     = CFEMStandardElementBase::GetNDOFsStatic(VTK_Type, nPolySol);
  elemIDGlobal = dataElem[4];

  /*--- Allocate the memory for the global nodes of the element to define
        the geometry and copy the data from dataElem. ---*/
  Nodes = new unsigned long[nDOFsGrid];
  for(unsigned short i=0; i<nDOFsGrid; ++i)
    Nodes[i] = dataElem[i+5];

  /*--- Determine the dimension of the element. ---*/
  nDim = (VTK_Type == TRIANGLE || VTK_Type == QUADRILATERAL) ? 2 : 3;

  /*--- Initialize the pointer members to nullptr. ---*/
  ElementOwnsFace        = nullptr;
  PeriodIndexNeighbors   = nullptr;
  JacobianFaceIsConstant = nullptr;
}

void CPrimalGridFEM::GetCornerPointsAllFaces(unsigned short &numFaces,
                                             unsigned short nPointsPerFace[],
                                             unsigned long  faceConn[6][4]) {

  /*--- Get the corner points of the faces local to the element. ---*/
  GetLocalCornerPointsAllFaces(VTK_Type, nPolyGrid, nDOFsGrid,
                               numFaces, nPointsPerFace, faceConn);

  /*--- Convert the local values of faceConn to global values. ---*/
  for(unsigned short i=0; i<numFaces; ++i) {
    for(unsigned short j=0; j<nPointsPerFace[i]; ++j) {
      unsigned long nn = faceConn[i][j];
      faceConn[i][j] = Nodes[nn];
    }
  }

  /*--- Store numFaces in nFaces for later purposes. ---*/
  nFaces = numFaces;
}

void CPrimalGridFEM::GetLocalCornerPointsAllFaces(unsigned short elementType,
                                                  unsigned short nPoly,
                                                  unsigned short nDOFs,
                                                  unsigned short &numFaces,
                                                  unsigned short nPointsPerFace[],
                                                  unsigned long  faceConn[6][4]) {

  /*--- Determine the element type and set the face data accordingly.
        The faceConn values are local to the element. ---*/
  unsigned short nn2, nn3, nn4;
  switch( elementType ) {
    case TRIANGLE:
      numFaces = 3;
      nPointsPerFace[0] = 2; faceConn[0][0] = 0;        faceConn[0][1] = nPoly;
      nPointsPerFace[1] = 2; faceConn[1][0] = nPoly;    faceConn[1][1] = nDOFs -1;
      nPointsPerFace[2] = 2; faceConn[2][0] = nDOFs -1; faceConn[2][1] = 0;
      break;

    case QUADRILATERAL:
      numFaces = 4; nn2 = nPoly*(nPoly+1);
      nPointsPerFace[0] = 2; faceConn[0][0] = 0;        faceConn[0][1] = nPoly;
      nPointsPerFace[1] = 2; faceConn[1][0] = nPoly;    faceConn[1][1] = nDOFs -1;
      nPointsPerFace[2] = 2; faceConn[2][0] = nDOFs -1; faceConn[2][1] = nn2;
      nPointsPerFace[3] = 2; faceConn[3][0] = nn2;      faceConn[3][1] = 0;
      break;

    case TETRAHEDRON:
      numFaces = 4; nn2 = (nPoly+1)*(nPoly+2)/2 -1; nn3 = nDOFs -1;
      nPointsPerFace[0] = 3; faceConn[0][0] = 0;     faceConn[0][1] = nPoly; faceConn[0][2] = nn2;
      nPointsPerFace[1] = 3; faceConn[1][0] = 0;     faceConn[1][1] = nn3;   faceConn[1][2] = nPoly;
      nPointsPerFace[2] = 3; faceConn[2][0] = 0;     faceConn[2][1] = nn2;   faceConn[2][2] = nn3;
      nPointsPerFace[3] = 3; faceConn[3][0] = nPoly; faceConn[3][1] = nn3;   faceConn[3][2] = nn2;
      break;

    case PYRAMID:
      numFaces = 5; nn2 = (nPoly+1)*(nPoly+1) -1; nn3 = nn2 - nPoly;
      nPointsPerFace[0] = 4; faceConn[0][0] = 0;     faceConn[0][1] = nPoly;    faceConn[0][2] = nn2; faceConn[0][3] = nn3;
      nPointsPerFace[1] = 3; faceConn[1][0] = 0;     faceConn[1][1] = nDOFs -1; faceConn[1][2] = nPoly;
      nPointsPerFace[2] = 3; faceConn[2][0] = nn3;   faceConn[2][1] = nn2;      faceConn[2][2] = nDOFs -1;
      nPointsPerFace[3] = 3; faceConn[3][0] = 0;     faceConn[3][1] = nn3;      faceConn[3][2] = nDOFs -1;
      nPointsPerFace[4] = 3; faceConn[4][0] = nPoly; faceConn[4][1] = nDOFs -1; faceConn[4][2] = nn2;
      break;

    case PRISM:
      numFaces = 5; nn2 = (nPoly+1)*(nPoly+2)/2; nn3 = nPoly*nn2; --nn2;
      nPointsPerFace[0] = 3; faceConn[0][0] = 0;     faceConn[0][1] = nPoly;     faceConn[0][2] = nn2;
      nPointsPerFace[1] = 3; faceConn[1][0] = nn3;   faceConn[1][1] = nn2+nn3;   faceConn[1][2] = nPoly+nn3;
      nPointsPerFace[2] = 4; faceConn[2][0] = 0;     faceConn[2][1] = nn3;       faceConn[2][2] = nPoly+nn3; faceConn[2][3] = nPoly;
      nPointsPerFace[3] = 4; faceConn[3][0] = 0;     faceConn[3][1] = nn2;       faceConn[3][2] = nn2+nn3;   faceConn[3][3] = nn3;
      nPointsPerFace[4] = 4; faceConn[4][0] = nPoly; faceConn[4][1] = nPoly+nn3; faceConn[4][2] = nn2+nn3;   faceConn[4][3] = nn2;
      break;

    case HEXAHEDRON:
      numFaces = 6; nn2 = (nPoly+1)*(nPoly+1); nn4 = nPoly*nn2; --nn2; nn3 = nn2 - nPoly;
      nPointsPerFace[0] = 4; faceConn[0][0] = 0;     faceConn[0][1] = nPoly;     faceConn[0][2] = nn2;       faceConn[0][3] = nn3;
      nPointsPerFace[1] = 4; faceConn[1][0] = nn4;   faceConn[1][1] = nn3+nn4;   faceConn[1][2] = nn2+nn4;   faceConn[1][3] = nPoly+nn4;
      nPointsPerFace[2] = 4; faceConn[2][0] = 0;     faceConn[2][1] = nn4;       faceConn[2][2] = nPoly+nn4; faceConn[2][3] = nPoly;
      nPointsPerFace[3] = 4; faceConn[3][0] = nn3;   faceConn[3][1] = nn2;       faceConn[3][2] = nn2+nn4;   faceConn[3][3] = nn3+nn4;
      nPointsPerFace[4] = 4; faceConn[4][0] = 0;     faceConn[4][1] = nn3;       faceConn[4][2] = nn3+nn4;   faceConn[4][3] = nn4;
      nPointsPerFace[5] = 4; faceConn[5][0] = nPoly; faceConn[5][1] = nPoly+nn4; faceConn[5][2] = nn2+nn4;   faceConn[5][3] = nn2;
      break;
  }
}

void CPrimalGridFEM::InitializeJacobianConstantFaces(unsigned short val_nFaces) {

  /*--- Allocate the memory for JacobianFaceIsConstant and initialize
        its values to false.     ---*/
  JacobianFaceIsConstant = new bool[val_nFaces];
  for(unsigned short i=0; i<val_nFaces; ++i)
    JacobianFaceIsConstant[i] = false;
}

void CPrimalGridFEM::InitializeNeighbors(unsigned short val_nFaces) {

  /*--- Allocate the memory for Neighbor_Elements and PeriodIndexNeighbors and
        initialize the arrays to -1 to indicate that no neighbor is present and
        that no periodic transformation is needed to the neighbor. ---*/
  Neighbor_Elements    = new long[val_nFaces];
  ElementOwnsFace      = new bool[val_nFaces];
  PeriodIndexNeighbors = new short[val_nFaces];

  for(unsigned short i=0; i<val_nFaces; ++i) {
    Neighbor_Elements[i]    = -1;
    ElementOwnsFace[i]      =  false;
    PeriodIndexNeighbors[i] = -1;
  }
}
