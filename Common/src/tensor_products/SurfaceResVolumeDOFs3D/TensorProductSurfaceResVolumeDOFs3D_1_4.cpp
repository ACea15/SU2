/*!
 * \file TensorProductSurfaceResVolumeDOFs3D_1_4.cpp
 * \brief Function, which carries out the tensor product for (nDOFs1D,nInt1D) = (1,4)
 * \author Automatically generated file, do not change manually
 * \version 7.1.1 "Blackbird"
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

#include "../../../include/tensor_products/TensorProductSurfaceResVolumeDOFs3D.hpp"
#include "../../../include/fem/CFEMStandardElementBase.hpp"

void TensorProductSurfaceResVolumeDOFs3D_1_4(const int           N,
                                             const int           faceID,
                                             const int           ldb,
                                             const int           ldc,
                                             const bool          swapTangDir,
                                             const passivedouble *An,
                                             const passivedouble *ATt0,
                                             const passivedouble *ATt1,
                                             const su2double     *B,
                                             su2double           *C) {

  /*--- Compute the padded value of the number of 1D DOFs. ---*/
  const int KP = CFEMStandardElementBase::PaddedValue(1);

  /*--- Cast the one dimensional input arrays for the tangential parts
        of the AT-tensor to 2D arrays. The normal part is a 1D array.
        Note that C++ stores multi-dimensional arrays in row major order,
        hence the indices are reversed compared to the column major order
        storage of e.g. Fortran. ---*/
  const passivedouble *an         = An;
  const passivedouble (*aTt0)[KP] = (const passivedouble (*)[KP]) ATt0;
  const passivedouble (*aTt1)[KP] = (const passivedouble (*)[KP]) ATt1;

  /*--- Define the variables to store the intermediate results. ---*/
  su2double bFace[1][KP];
  su2double tmpI[4][KP], tmpJ[4][4];

  /*--- Outer loop over N. ---*/
  for(int l=0; l<N; ++l) {
    const su2double (*b)[4] = (const su2double (*)[4]) &B[l*ldb];
    su2double (*c)[1][1] = (su2double (*)[1][1]) &C[l*ldc];

    /*--- Copy the value from the appropriate location in c.
          Take a possible swapping into account. ---*/
    if( swapTangDir ) {
      for(int j=0; j<4; ++j)
        for(int i=0; i<4; ++i)
          tmpJ[j][i] = b[j][i];
    }
    else {
      for(int j=0; j<4; ++j)
        for(int i=0; i<4; ++i)
          tmpJ[j][i] = b[i][j];
    }

    /*--- Tensor product in second tangential direction to obtain the data
          in the DOFs in this direction of the face. ---*/
    for(int i=0; i<4; ++i) {
      SU2_OMP_SIMD
      for(int j=0; j<KP; ++j) tmpI[i][j] = 0.0;
      for(int jj=0; jj<4; ++jj) {
        SU2_OMP_SIMD_IF_NOT_AD
        for(int j=0; j<KP; ++j)
          tmpI[i][j] += aTt1[jj][j] * tmpJ[jj][i];
      }
    }

    /*--- Tensor product in first tangential direction to obtain the data
          in the DOFs in the both direction of the face. ---*/
    for(int j=0; j<1; ++j) {
      SU2_OMP_SIMD
      for(int i=0; i<KP; ++i) bFace[j][i] = 0.0;
      for(int ii=0; ii<4; ++ii) {
        SU2_OMP_SIMD_IF_NOT_AD
        for(int i=0; i<KP; ++i)
          bFace[j][i] += aTt0[ii][i] * tmpI[ii][j];
      }
    }

    /*--- Tensor product in normal direction to update the residual. ---*/
    c[0][0][0] += an[0]*bFace[0][0];

  } /*--- End of the loop over N. ---*/
}
