/*!
 * \file GenerateSourceFiles.cpp
 * \brief Program that creates the source files for the desired
 *        combinations for the tensor products to compute the
 *        data in the 2D volume integration points.
 * \author E. van der Weide
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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

/*----------------------------------------------------------------------------*/
/* This program creates the source files to compute the data in the 2D        */
/* integration points of a quadrilateral using tensor products. The number    */
/* of 1D DOFs and integration points are hard coded, such that the compiler   */
/* can optimize much better. The desired combinations of DOFs and integration */
/* points in 1D are specified below. For each combination a separate source   */
/* file is created, while the function headers are stored in a single include */
/* file. Also the meson.build file is created. The names of the functions are */
/* TensorProductVolumeIntPoints2D_K_M, where K is the number of DOFs in 1D    */
/* M the number of integration points in 1D. These functions are stored in    */
/* the files TensorProductVolumeIntPoints2D_K_M.cpp, while the function       */
/* headers are stored in TensorProductVolumeIntPoints2D.hpp in the directory  */
/* ../../../include/tensor_products.                                          */
/*----------------------------------------------------------------------------*/

/*--- Define the string, which contains the combinations of number of DOFs
      and integration points in 1D for which the code must be generated for
      the 2D tensor product. The convention is as follows:
      K1-M1_K2-M2_K3-M3_....., where K1 and M1 are the number of 1D DOFs and
      1D integration points, respectively, for the 1st pair, K2 and M2 for
      the second pair, etc. So the number of DOFs and integration points are
      separated by - and the pairs by a _. ---*/
const std::string Pairs = "1-2_1-3_1-4_1-5_2-2_2-3_2-4_2-5_3-3_3-4_3-5_3-6_"
                          "3-7_3-8_4-4_4-5_4-6_4-7_4-8_5-5_5-6_5-7_5-8_6-6_"
                          "6-7_6-8_6-9_7-7_7-8_7-9_7_12_8-8_8-12_8-13_9-9_"
                          "9-13_9-14_10-10_10-14";

/*--- Define the name of the include file. ---*/
const std::string IncludeDir  = "../../../include/tensor_products";
const std::string IncludeFile = "TensorProductVolumeIntPoints2D.hpp";

/*--- Define the name of the include file for the standard element. ---*/
const std::string StandardElemIncludeFile = "../../../include/fem/CFEMStandardElementBase.hpp";

/*----------------------------------------------------------------------------*/
/*                          Function prototypes.                              */
/*----------------------------------------------------------------------------*/

void CreateIncludeFile(const std::vector<int> &nDOFs1D,
                       const std::vector<int> &nInt1D);

void CreateMapSourceFile(const std::vector<int> &nDOFs1D,
                         const std::vector<int> &nInt1D);

void CreateMesonBuildFile(const std::vector<int> &nDOFs1D,
                          const std::vector<int> &nInt1D);

void CreateTensorProductSourceFile(const int nDOFs1D,
                                   const int nInt1D);

void ExtractPairsFromString(std::vector<int> &nDOFs1D,
                            std::vector<int> &nInt1D);

void StringReadErrorHandler(const size_t ind);

void WriteFileHeader(std::ofstream &file,
                     const char    *fileName,
                     const char    *description);

/*----------------------------------------------------------------------------*/
/*                              Main program.                                 */
/*----------------------------------------------------------------------------*/

int main() {

  /* Extract the pairing information from the global string Pairs to integer data. */
  std::vector<int> nDOFs1D, nInt1D;
  ExtractPairsFromString(nDOFs1D, nInt1D);

  /* Create the include file with the function prototypes. */
  CreateIncludeFile(nDOFs1D, nInt1D);

  /* Create the meson build file. */
  CreateMesonBuildFile(nDOFs1D, nInt1D);

  /* Create the source file for CreateMapTensorProductVolumeIntPoints3D. */
  CreateMapSourceFile(nDOFs1D, nInt1D);

  /* Loop to create the source files that carry out the actual tensor product. */
  for(unsigned int i=0; i<nDOFs1D.size(); ++i)
    CreateTensorProductSourceFile(nDOFs1D[i], nInt1D[i]);

  /* Return zero to indicate that everything went fine. */
  return 0;
}

/*----------------------------------------------------------------------------*/
/*                      Implementation of the functions.                      */
/*----------------------------------------------------------------------------*/

/* Function, which creates the include file with the function prototypes. */
void CreateIncludeFile(const std::vector<int> &nDOFs1D,
                       const std::vector<int> &nInt1D) {

  /* Create the name of the include file, including the relative path. */
  const std::string fileName = IncludeDir + "/" + IncludeFile;

  /* Open the include file for writing and check if it went OK. */
  std::ofstream includeFile(fileName.c_str());
  if( !includeFile ) {
    std::cout << std::endl;
    std::cout << "The file " << fileName << std::endl
              << "could not be opened for writing." << std::endl;
    std::exit(1);
  }

  /* Write the header of the file. */
  WriteFileHeader(includeFile, IncludeFile.c_str(),
                  "Function prototypes for the tensor product to compute "
                  "the data in the 2D integration points");

  /* Write the pragma once line and the required include files. */
  includeFile << "#pragma once" << std::endl << std::endl;

  includeFile << "#include <iostream>" << std::endl;
  includeFile << "#include <map>"   << std::endl;

  includeFile << "#include \"../basic_types/datatype_structure.hpp\"" << std::endl;
  includeFile << "#include \"../parallelization/omp_structure.hpp\"" << std::endl;
  includeFile << "#include \"../toolboxes/classes_multiple_integers.hpp\"" << std::endl;
  includeFile << std::endl;

  /* Write that namespace std is used. */
  includeFile << "using namespace std;" << std::endl << std::endl;

  /* Typedef the function pointer for the tensor product. */
  includeFile << "typedef void(*TPI2D)(const int           N,"   << std::endl;
  includeFile << "                     const int           ldb," << std::endl;
  includeFile << "                     const int           ldc," << std::endl;
  includeFile << "                     const passivedouble *Ai," << std::endl;
  includeFile << "                     const passivedouble *Aj," << std::endl;
  includeFile << "                     const su2double     *B,"  << std::endl;
  includeFile << "                     su2double           *C);" << std::endl;

  /* Write the function prototype to create the map of function pointers
     of the available tensor product functions. */
  includeFile << "/*!" << std::endl;
  includeFile << " * \\brief Function, which stores the available function pointers for the tensor" << std::endl;
  includeFile << " *        product for the 2D volume integration points in a map." << std::endl;
  includeFile << " * \\param[out] mapFunctions - Map to store the function pointers to carry out the tensor product." << std::endl;
  includeFile << " */" << std::endl;
  includeFile << "void CreateMapTensorProductVolumeIntPoints2D(map<CUnsignedShort2T, TPI2D> &mapFunctions);" << std::endl;

  /* Loop over the number of combinations for which the tensor product must be created. */
  for(unsigned int i=0; i<nDOFs1D.size(); ++i) {

    /* Determine the name of this function. */
    std::ostringstream functionName;
    functionName << "TensorProductVolumeIntPoints2D_"
                 << nDOFs1D[i] << "_" << nInt1D[i];

    /* Write the function prototype for this tensor product. */
    includeFile << std::endl;
    includeFile << "/*!" << std::endl;
    includeFile << " * \\brief Function, which carries out the tensor product to obtain the data" << std::endl;
    includeFile << " *        in the 2D integration points for (nDOFs1D,nInt1D) = ("
                << nDOFs1D[i] << "," << nInt1D[i] << ")." << std::endl;
    includeFile << " * \\param[in]  N   - Number of variables to be determined in the integration points" << std::endl;
    includeFile << " * \\param[in]  ldb - Leading dimension of B when stored as a matrix." << std::endl;
    includeFile << " * \\param[in]  ldc - Leading dimension of C when stored as a matrix." << std::endl;
    includeFile << " * \\param[in]  Ai  - I-componnent of the A tensor." << std::endl;
    includeFile << " * \\param[in]  Aj  - J-componnent of the A tensor." << std::endl;
    includeFile << " * \\param[in]  B   - Tensor, which contains the data to be interpolated." << std::endl;
    includeFile << " * \\param[out] C   - Result of the tensor product C = A*B." << std::endl;
    includeFile << " */" << std::endl;
    includeFile << "void " << functionName.str() << "(const int           N,"   << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "const int           ldb,"   << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "const int           ldc,"   << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "const passivedouble *Ai," << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "const passivedouble *Aj," << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "const su2double     *B," << std::endl;

    for(unsigned int j=0; j<(functionName.str().size()+6); ++j) includeFile << " ";
    includeFile << "su2double           *C);" << std::endl;
  }

  /* Close the include file again. */
  includeFile.close();
}

/* Function, which creates the source code for the function
   CreateMapTensorProductVolumeIntPoints2D. */
void CreateMapSourceFile(const std::vector<int> &nDOFs1D,
                         const std::vector<int> &nInt1D) {

  /* Open the source file for writing and check if it went OK. */
  std::ofstream sourceFile("CreateMapTensorProductVolumeIntPoints2D.cpp");
  if( !sourceFile ) {
    std::cout << std::endl;
    std::cout << "The file CreateMapTensorProductVolumeIntPoints2D.cpp "
              << "could not be opened for writing." << std::endl;
    std::exit(1);
  }

  /* Write the header of the file. */
  WriteFileHeader(sourceFile, "CreateMapTensorProductVolumeIntPoints2D.cpp",
                  "Function, which creates the map between the number of 1D DOFs "
                  "and integration points and the function pointers.");

  /* Write the line for the include file. */
  sourceFile << "#include \"" << IncludeDir << "/" << IncludeFile << "\"" << std::endl
             << std::endl;

  /* Write the header of the function. */
  sourceFile << "void CreateMapTensorProductVolumeIntPoints2D(map<CUnsignedShort2T, TPI2D> &mapFunctions) {" << std::endl;

  /* Make sure that the map is empty. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Make sure that the map is empty. ---*/" << std::endl;
  sourceFile << "  mapFunctions.clear();" << std::endl;

  /* Define the CUnsignedShort2T, which will store the number of 1D DOFs
     and integration points. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Variable to store the number of DOFs and integration points as one entity. ---*/" << std::endl;
  sourceFile << "  CUnsignedShort2T nDOFsAndInt;" << std::endl;

  /* Write the comment line. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Insert the mappings from the CUnsignedShort2T to the function pointer. ---*/" << std::endl;

  /* Loop over the tensor product functions that are available. */
  for(unsigned int i=0; i<nDOFs1D.size(); ++i) {

    /* Determine the name of this function. */
    std::ostringstream functionName;
    functionName << "TensorProductVolumeIntPoints2D_"
                 << nDOFs1D[i] << "_" << nInt1D[i];

    /* Write a new line, if needed, and set the entries of nDOFsAndInt. */
    if( i ) sourceFile << std::endl;
    sourceFile << "  nDOFsAndInt.short0 = " << nDOFs1D[i]
               << "; nDOFsAndInt.short1 = " << nInt1D[i] << ";" << std::endl;

    /* Create the map entry. */
    sourceFile << "  mapFunctions.emplace(nDOFsAndInt, &" << functionName.str() << ");" << std::endl;
  }

  /* Write the closing bracket of this function and close the file again. */
  sourceFile << "}" << std::endl;
  sourceFile.close();
}

/* Function, which writes the meson build file. */
void CreateMesonBuildFile(const std::vector<int> &nDOFs1D,
                          const std::vector<int> &nInt1D) {

  /* Open the meson build file for writing and check if it went OK. */
  std::ofstream mesonFile("meson.build");
  if( !mesonFile ) {
    std::cout << std::endl;
    std::cout << "The file meson.build could not be opened for writing." << std::endl;
    std::exit(1);
  }

  /* Write the file, which indicates which files must be compiled in this directory. */
  mesonFile << "common_src += files([";

  /* Loop over the number of combinations for which a tensor product
     function must be generated. */
  for(unsigned int i=0; i<nDOFs1D.size(); ++i) {

    /* Determine the name of this file and write it to the meson file. */
    std::ostringstream fileName;
    fileName << "TensorProductVolumeIntPoints2D_"
             << nDOFs1D[i] << "_" << nInt1D[i] << ".cpp";
    mesonFile << "'" << fileName.str() << "'," << std::endl;

    /* Write the number of blanks for the next file name. */
    mesonFile << "                     ";
  }

  /* Write the name of the final source file, which is the file
     to create the map for the function pointers. */
  mesonFile << "'CreateMapTensorProductVolumeIntPoints2D.cpp'])" << std::endl;

  /* Close the file again. */
  mesonFile.close();
}

/* Function, which creates the source code for the tensor product
   for the given arguments. */
void CreateTensorProductSourceFile(const int nDOFs1D,
                                   const int nInt1D) {

  /* Determine the name of this function and open the corresponding file name. */
  std::ostringstream functionName;
  functionName << "TensorProductVolumeIntPoints2D_"
               << nDOFs1D << "_" << nInt1D;
  const std::string fileName = functionName.str() + ".cpp";

  /* Open the file for writing and check if it went OK. */
  std::ofstream sourceFile(fileName.c_str());
  if( !sourceFile ) {
    std::cout << std::endl;
    std::cout << "The file " << fileName
              << " could not be opened for writing." << std::endl;
    std::exit(1);
  }

  /* Write the header of the file. */
  std::ostringstream description;
  description << "Function, which carries out the tensor product for (nDOFs1D,nInt1D) = ("
              << nDOFs1D << "," << nInt1D << ")";
  WriteFileHeader(sourceFile, fileName.c_str(), description.str().c_str());

  /* Write the lines for the include file. */
  sourceFile << "#include \"" << IncludeDir << "/" << IncludeFile << "\"" << std::endl;
  sourceFile << "#include \"" << StandardElemIncludeFile << "\"" << std::endl
             << std::endl;

  /* Write the header of the function. */
  sourceFile << "void " << functionName.str() << "(const int           N,"   << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "const int           ldb,"   << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "const int           ldc,"   << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "const passivedouble *Ai," << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "const passivedouble *Aj," << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "const su2double     *B," << std::endl;

  for(unsigned int j=0; j<(functionName.str().size()+6); ++j) sourceFile << " ";
  sourceFile << "su2double           *C) {" << std::endl;

  /* Compute the padded value of the number of integration points. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Compute the padded value of the number of integration points. ---*/" << std::endl;
  sourceFile << "  const int MP = CFEMStandardElementBase::PaddedValue(" << nInt1D << ");" << std::endl;

  /* Cast the components of the A tensor to a 2D array. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Cast the one dimensional input arrays for the A-tensor to 2D arrays." << std::endl;
  sourceFile << "        Note that C++ stores multi-dimensional arrays in row major order,"    << std::endl;
  sourceFile << "        hence the indices are reversed compared to the column major order"    << std::endl;
  sourceFile << "        storage of e.g. Fortran. ---*/"                                       << std::endl;
  sourceFile << "  const passivedouble (*ai)[MP] = (const passivedouble (*)[MP]) Ai;" << std::endl;
  sourceFile << "  const passivedouble (*aj)[MP] = (const passivedouble (*)[MP]) Aj;" << std::endl;

  /* Define the variables to store the intermediate results. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Define the variables to store the intermediate results. ---*/" << std::endl;
  sourceFile << "  su2double tmpJ[" << nDOFs1D << "][MP];" << std::endl;

  /* Start the outer loop over N. */
  sourceFile << std::endl;
  sourceFile << "  /*--- Outer loop over N. ---*/" << std::endl;
  sourceFile << "  for(int l=0; l<N; ++l) {" << std::endl;

  /* Cast the index l of B and C to multi-dimensional arrays. */
  sourceFile << std::endl;
  sourceFile << "    /*--- Cast the index l of B and C to multi-dimensional arrays. ---*/" << std::endl;
  sourceFile << "    const su2double (*b)[" << nDOFs1D << "] = "
             << "(const su2double (*)[" << nDOFs1D << "]) &B[l*ldb];" << std::endl;
  sourceFile << "    su2double       (*c)[" << nInt1D << "] = "
             << "(su2double (*)[" << nInt1D << "]) &C[l*ldc];" << std::endl;

  /* Tensor product in j-direction. */
  sourceFile << std::endl;
  sourceFile << "    /*--- Tensor product in j-direction to obtain the solution" << std::endl;
  sourceFile << "          in the integration points in j-direction. ---*/"      << std::endl;
  sourceFile << "    for(int i=0; i<" << nDOFs1D << "; ++i) {" << std::endl;
  sourceFile << "      SU2_OMP_SIMD" << std::endl;
  sourceFile << "      for(int j=0; j<MP; ++j) tmpJ[i][j] = 0.0;" << std::endl;
  sourceFile << "      for(int jj=0; jj<" << nDOFs1D << "; ++jj) {" << std::endl;
  sourceFile << "        SU2_OMP_SIMD_IF_NOT_AD" << std::endl;
  sourceFile << "        for(int j=0; j<MP; ++j)" << std::endl;
  sourceFile << "          tmpJ[i][j] += aj[jj][j] * b[jj][i];" << std::endl;
  sourceFile << "      }" << std::endl;
  sourceFile << "    }" << std::endl;

  /* Tensor product in i-direction. */
  sourceFile << std::endl;
  sourceFile << "    /*--- Check if the padded number MP is larger than the" << std::endl;
  sourceFile << "          number the integration points. In this case the"  << std::endl;
  sourceFile << "          result cannot be stored directly in c. ---*/"     << std::endl;
  sourceFile << "    if(MP > " <<  nInt1D << ") {" << std::endl << std::endl;

  sourceFile << "    /*--- Tensor product in i-direction to obtain the solution" << std::endl;
  sourceFile << "          in the integration points in i-direction. This is"    << std::endl;
  sourceFile << "          the final result of the tensor product, which is"     << std::endl;
  sourceFile << "          in tmpI for performance reasons. ---*/"               << std::endl;
  sourceFile << "      su2double tmpI[MP];" << std::endl;
  sourceFile << "      for(int j=0; j<" << nInt1D << "; ++j) {" << std::endl;
  sourceFile << "        SU2_OMP_SIMD" << std::endl;
  sourceFile << "        for(int i=0; i<MP; ++i) tmpI[i] = 0.0;" << std::endl;
  sourceFile << "        for(int ii=0; ii<" << nDOFs1D << "; ++ii) {" << std::endl;
  sourceFile << "          SU2_OMP_SIMD_IF_NOT_AD" << std::endl;
  sourceFile << "          for(int i=0; i<MP; ++i)" << std::endl;
  sourceFile << "            tmpI[i] += ai[ii][i] * tmpJ[ii][j];" << std::endl;
  sourceFile << "        }" << std::endl << std::endl;

  sourceFile << "        /*--- Copy the values to the appropriate location in c. ---*/" << std::endl;
  sourceFile << "        for(int i=0; i<" << nInt1D << "; ++i)" << std::endl;
  sourceFile << "          c[j][i] = tmpI[i];" << std::endl;
  sourceFile << "      }" << std::endl;
  sourceFile << "    }" << std::endl;
  sourceFile << "    else {" << std::endl << std::endl;

  sourceFile << "      /*--- The padded number equals the number of integration"        << std::endl;
  sourceFile << "            points. The result of the tensor product in"               << std::endl;
  sourceFile << "            i-direction to obtain the solution in the integration"     << std::endl;
  sourceFile << "            points in i-direction can be stored in c directly. ---*/" << std::endl;
  sourceFile << "      for(int j=0; j<" << nInt1D << "; ++j) {" << std::endl;
  sourceFile << "        SU2_OMP_SIMD" << std::endl;
  sourceFile << "        for(int i=0; i<MP; ++i) c[j][i] = 0.0;" << std::endl;
  sourceFile << "        for(int ii=0; ii<" << nDOFs1D << "; ++ii) {" << std::endl;
  sourceFile << "          SU2_OMP_SIMD_IF_NOT_AD" << std::endl;
  sourceFile << "          for(int i=0; i<MP; ++i)" << std::endl;
  sourceFile << "            c[j][i] += ai[ii][i] * tmpJ[ii][j];" << std::endl;
  sourceFile << "        }" << std::endl;
  sourceFile << "      }" << std::endl;
  sourceFile << "    }" << std::endl;

  /* Close the outer loop over N. */
  sourceFile << std::endl;
  sourceFile << "  } /*--- End of the loop over N. ---*/" << std::endl;

  /* Write the closing bracket of this function and close the file again. */
  sourceFile << "}" << std::endl;
  sourceFile.close();
}

/* Function, which extracts the integer pairs from the global string Pairs. */
void ExtractPairsFromString(std::vector<int> &nDOFs1D,
                            std::vector<int> &nInt1D) {

  /* While loop to extract the pairing information from the global string Pairs. */
  size_t ind = 0;
  while(ind < Pairs.size()) {

    /* Locate the next occurance of "-" in the string. This data must be found. */
    size_t indNew = Pairs.find("-", ind);
    if(indNew == std::string::npos) StringReadErrorHandler(ind);

    /* Extract the information for the number of DOFs. */
    int data;
    std::string nDOFsString = Pairs.substr(ind, indNew-ind);
    std::istringstream istrNDOFs(nDOFsString);
    if( !(istrNDOFs >> data) ) StringReadErrorHandler(ind);

    nDOFs1D.push_back(data);

    /* Update ind to one step after indNew and check if it is still valid. */
    ind = indNew+1;
    if(ind >= Pairs.size()) StringReadErrorHandler(ind);

    /* Locate the next occurance of "_" in the string. If it is not found,
       set indNew to the size of Pairs. */
    indNew = Pairs.find("_", ind);
    if(indNew == std::string::npos) indNew = Pairs.size();

    /* Extract the information for the number of integration points. */
    std::string nIntsString = Pairs.substr(ind, indNew-ind);
    std::istringstream istrNInts(nIntsString);
    if( !(istrNInts >> data) ) StringReadErrorHandler(ind);

    nInt1D.push_back(data);

    /* Update ind to one step after indNew. */
    ind = indNew+1;
  }
}

/*----------------------------------------------------------------------------*/

/* Error handler when something goes wrong when reading the string. */
void StringReadErrorHandler(const size_t ind) {

  /* Determine the indices in the string for the error message. */
  size_t indStart = 0;
  if(ind >= 4) indStart = ind - 4;

  size_t indEnd = ind + 4;
  if(indEnd > Pairs.size()) indEnd = Pairs.size();

  /* Write the error message. */
  std::cout << std::endl;
  std::cout << "Something goes wrong when reading the global string Pairs." << std::endl;
  std::cout << "Approximate part of the string where it goes wrong: "
            << Pairs.substr(indStart, indEnd-indStart) << std::endl;

  /* Exit the program. */
  std::exit(1);
}

/*----------------------------------------------------------------------------*/

/* Function, which writes the header of the file. */
void WriteFileHeader(std::ofstream &file,
                     const char    *fileName,
                     const char    *description) {

  /* Write the name of the file and the description. */
  file << "/*!" << std::endl
       << " * \\file " << fileName << std::endl
       << " * \\brief " << description << std::endl;

  /* Write the author. To indicate that it is an automatically generated file
     set the author name to automatically generated, do not change manually. */
  file << " * \\author Automatically generated file, do not change manually" << std::endl;

  /* Write the remainder of the header file. */
  file << " * \\version 7.1.1 \"Blackbird\"" << std::endl;
  file << " *" << std::endl;
  file << " * SU2 Project Website: https://su2code.github.io" << std::endl;
  file << " *" << std::endl;
  file << " * The SU2 Project is maintained by the SU2 Foundation" << std::endl;
  file << " * (http://su2foundation.org)" << std::endl;
  file << " *" << std::endl;
  file << " * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)" << std::endl;
  file << " *" << std::endl;
  file << " * SU2 is free software; you can redistribute it and/or" << std::endl;
  file << " * modify it under the terms of the GNU Lesser General Public" << std::endl;
  file << " * License as published by the Free Software Foundation; either" << std::endl;
  file << " * version 2.1 of the License, or (at your option) any later version." << std::endl;
  file << " *" << std::endl;
  file << " * SU2 is distributed in the hope that it will be useful," << std::endl;
  file << " * but WITHOUT ANY WARRANTY; without even the implied warranty of" << std::endl;
  file << " * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU" << std::endl;
  file << " * Lesser General Public License for more details." << std::endl;
  file << " *" << std::endl;
  file << " * You should have received a copy of the GNU Lesser General Public" << std::endl;
  file << " * License along with SU2. If not, see <http://www.gnu.org/licenses/>." << std::endl;
  file << " */" << std::endl;
  file << std::endl;
}
