step 1: step-44-firstPiola.cc 
        tau_vol               ...          Done
        tau_bar -> tau_iso    ...          Done
        Jc_vol                ...          Done (typo in step-44 Jc_vol  p_hat)
        Jc_iso                ...          Done
        P_vol                 ...          Done
        P_bar -> P_iso        ...          Done
        A_vol                 ...
        A_iso                 ...
        ***confusing.
        
        For deviatoric tensor: https://en.wikipedia.org/wiki/Cauchy_stress_tensor#Stress_deviator_tensor
        For F non-symmetric, dF/dF = II (4th order identity tensor)
        For C symmetric,     dC/dC = IIsym (4th order symmetric identity tensor)[2]           
        For F non-symmetric, d(Finv)/dF = II_Finv (4th order tensor based on Finverse) 
                                          II_Finv[ijkl] = - Finv[ik] * Finv[jl]    [1]
        For C symmetric,     d(Cinv)/dC = IIsym_Cinv (..............Cinverse) 
                                          IIsym_Cinv[ijkl] = 1/2(Cinv[ik] * Cinv[jl] + Cinv[il] * Cinv[kj])
        
        dPhi/dF = 2* dPhi/db * F [1]
        db_bar/db = J^(-2/3)*deviatoric tensor (...   )
        dF_bar/dF = J^(-1/3)*deviatoric tensor (...   )
        chain rule: d(Phi * B)/dF = Phi * dB/dF + outer_product(B, dPhi/dF)
        Also see p29 p35 [3] for many formulas (many typos)
        
        step-44-secondPiola.cc
        For stress and elasticity-tensor, use invariants as intermediate variables[4]
        push-forward of elasticity tensor  c_spatial[IJKL] = F[Ii]F[Jj]F[Kk]F[Ll]Cmaterial[ijkl]
        
        For assembling system tanget and rhs
        AB:C = transpose(C)*A:transpose(B) = B*transpose(C):transpose(A)
        Another form in [5]
        
References:
1)http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_III/Chapter_1_Vectors_Tensors/Vectors_Tensors_15_Tensor_Calculus_2.pdf
2)http://www.ce.berkeley.edu/~sanjay/ce231mse211/symidentity.pdf
3)Comella thesis.
4)Mechanics of Deformable Solids
5)Finite Element Formulations for large Deformation Dynamic Analysis

step 2: step-44-growth.cc
        needs to rework everything. All derivations are w.r.t F_e, but then transformation between configurations are through F.
        ***confusing.
step 3: step-38-surface-core-firstPiola.cc and step-38-surface-core-Kirchhoff.cc

