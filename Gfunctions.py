

# #print("rs,mass,temp,ssi:",rs,mass,temp,ssi)
# #print("mscaled,Tscaled,Siscaled,Gsph",mscaled,Tscaled,Siscaled,Gsph)
# # print("X shape:",X.shape)
#
# #Gi = self.newG.predict(X,index=self.eqnidx)*1e-9
# if self.eqnidx == 8:
#     # Real_weak_0500
#     Gi =(((Gscaled** 1.3153063) / (((mscaled / 1.1682062) + 2.6606467) / mscaled)) +0.1123054)
# elif (self.eqnidx == 5):
#     #Gi = Gscaled/((3.4193382/(mscaled+(1.226832*Gscaled)))+0.60731494)
#     #Gi = (Gscaled+(mscaled*0.007928836))/((1.1189142-(Gscaled*0.16170673))+(1.5297577/mscaled))
#     #Gi = (Gscaled + 0.088217504)/(((-1*Gscaled*0.17781049) + (1.8599294/mscaled))+1.1031982)
#     #Gi = (Gscaled / (((3.7769527 / mscaled) / Gscaled)**(0.5)+0.5059553)) + 0.10548777
#     #Gi = (mscaled/((((RHscaled/1.4598137)+Tscaled)+(Tscaled*((Tscaled*0.53852797)*mscaled)))/0.53032446))-(0.029204821/(1.3395258))
#     Gi = (((Gscaled*((rscaled-0.095324)*0.35078))-(0.84341-Gscaled)**3)/(((((rscaled*0.018373)-RHscaled)*Gscaled)*0.74768)+5.5581))+0.10616
# elif (self.eqnidx == 4):
#     Gi = ((mscaled * 0.1286035)**(1/4) * Gscaled) - 0.006103022
# elif (self.eqnidx == 7):
#     # Real_weak_0500
#     Gi =Gscaled / (0.6148354 - 1/(((Gscaled * -0.9017731) - mscaled ) / 3.130833))
# elif (self.eqnidx == 9):
#     # Real_weak_5000_excluded
#     Gi = 1/((((1.6862676/mscaled)**3+1.2213485)-(Gscaled*0.2888407)**3)/Gscaled)+0.058938276
# elif (self.eqnidx == 12):
#     # Real_weak_5000
#     Gi = (Gscaled*(RHscaled+0.60490817))/(((RHscaled+(1.3078504/mscaled)**2)+0.66782004)**Gscaled) # 10
#     #Gi = (Gscaled*(RHscaled + 0.59379333))/(((-0.24833941/mscaled)+((RHscaled+0.6751641)+(-1.5232717/mscaled)**2))**Gscaled)
# else:
#     # Real_weak_0500 (eqn. 11)
#     #'inv((((2.030117 / Gc) ^ 0.7968155) - 0.3890275) * (((2.8517783 / (m_scaled - -2.0615888)) + -0.06024278) - -0.5617665))'
#     Gi = ((((2.030117/Gscaled)**0.7968155)-0.3890275)*(((2.8517783/(mscaled+2.0615888))-0.06024278)+0.5617665))**(-1)
# Gi = Gi*1e-9
# #Gi = Gi*Gsph
# #print(Gi,Gsph)