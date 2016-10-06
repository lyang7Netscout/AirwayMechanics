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
        dF/dF = II (4th order identity tensor)
        d(Finv)/dF = II_Finv (4th order tensor based on Finverse)
        dPhi/dF = 2* dPhi/db * F
        db_bar/db = J^(-2/3)*deviatoric tensor (...   )
        dF_bar/dF = J^(-1/3)*deviatoric tensor (...   )
        chain rule: d(Phi * B)/dF = Phi * dB/dF + outer_product(B, dPhi/dF)
        Also see p29 p35 Ref1 for many formulas (many typos)
        
step 2: step-44-growth.cc
        needs to rework everything. All derivations are w.r.t F_e, but then transformation between configurations are through F.
        ***confusing.
step 3: step-38-surface-core-firstPiola.cc and step-38-surface-core-Kirchhoff.cc

