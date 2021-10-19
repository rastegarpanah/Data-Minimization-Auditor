import pandas as pd
import numpy as np
from scipy.stats import beta
import random
import sys
from sklearn.metrics import accuracy_score
from numpy.random import Generator, PCG64

def feature_importance_categorical(clf, categorical_names):    
    index_tuples = zip([f.split('_')[0] for f in categorical_names],categorical_names)
    indices = pd.MultiIndex.from_tuples(index_tuples)
    return pd.Series(clf.feature_importances_, index =indices)

class RedundantModel():
    
    def __init__(self, clf, real_features):
        self.clf = clf
        self.F = real_features
    
    def predict(self, dataframe):
        return self.clf.predict(dataframe[self.F])


class Auditor():
    
    def __init__(self, clf, D, prior=(0.5,0.5)):
        self.clf = clf
        self.prior = prior
        self.D = D.copy()
        
        if (self.D.dtypes == 'category').any():
            self.is_categorical = True
            self.D_dummies = pd.get_dummies(self.D)
            self.Y = pd.Series(self.clf.predict(self.D_dummies), index=self.D.index)
        else:
            self.is_categorical = False
            self.Y = pd.Series(self.clf.predict(self.D), index=self.D.index)
                
        self.features = self.D.columns.tolist()
        self.domain = {}
        for f in self.features:
            self.domain[f] = sorted(self.D[f].unique().tolist())
            
        index_tuples = []
        for f in self.features:
            l = [(f,b) for b in self.domain[f]]
            index_tuples += l            
        self.arms = pd.MultiIndex.from_tuples(index_tuples, names=['feature', 'value'])
        self.init_counters()
        return
    
    def init_counters(self, scaled=False):
        self.S = pd.Series(0, index = self.arms, name='S')
        self.F = pd.Series(0, index = self.arms, name='F')
        self.scales = pd.Series(1.0, index = self.arms, name='scales')
        self.is_scaled = scaled
        if self.is_scaled:
            self.compute_scales()
        return
    
    def compute_scales(self):
        #compute prior scales from data
        for f in self.features:
            value_counts = self.D[f].value_counts(normalize=True)
            gammas = 1 - value_counts
            for b in self.domain[f]:
                self.scales[(f,b)] = gammas[b]
        #check for zero sensitivity in data
        if (self.scales==0).any():
            print('There is no variability in the audit data for the following features:')
            print(self.scales[self.scales==0])
            print('Data minimization level is 0.\n')
            raise SystemExit()
        #compute data subsets
        subset_indices = {}
        for f in self.features:
            for b in self.domain[f]:
                subset_indices[(f,b)] = self.D.index[self.D[f] != b]
        self.subset_indices = pd.Series(subset_indices, index = self.arms)
        return
        
    def impute_df(self,S,b):
        X = self.D.copy()
        X[S] = b
        if not isinstance(S, list):
            S = [S]
        for f in S:
            if self.D[f].dtype.name == 'category':
                X = X.astype({f:self.D[f].dtype})
                if X[f].isnull().any():
                    sys.exit('Error: imputation value not in the population domain')
        return X
    
    def impute_slow(self,ID,f,b):#slower than impute(), still useful if we can only sample from data dist online.
        x = self.D.loc[[ID]].copy()
        x[f] = b
        if self.is_categorical:
            if self.D[f].dtype.name == 'category':
                x = x.astype({f:self.D[f].dtype})
                if x[f].isnull().any():
                    sys.exit('Error: imputation value not in the domain')
            x = pd.get_dummies(x)
        return x
        
    def impute(self,ID,f,b):
        if self.is_categorical:
            x = self.D_dummies.loc[[ID]].copy()
            if self.D[f].dtype.name == 'category':
                for c in self.D[f].cat.categories:
                    x['%s_%s'%(f,c)] = 0
                x['%s_%s'%(f,b)] = 1
            else:
                x[f] = b
        else:
            x = self.D.loc[[ID]].copy()
            x[f] = b
        return x
    
    def instability_df_single_feature(self, f):
        imputation_instability = {}
        for b in self.domain[f]:
            D_tmp = self.impute_df(f,b)
            if self.is_categorical:
                D_tmp = pd.get_dummies(D_tmp)
            Y_imputed = self.clf.predict(D_tmp)
            imputation_instability[b] = 1.0 - accuracy_score(self.Y, Y_imputed)
        imputation_instability = pd.Series(imputation_instability)
        return {'imputation':imputation_instability.idxmin(), 'beta':imputation_instability.min()}
        
    def instability_df(self):
        feature_instability = {}
        for f in self.D.columns:
            feature_instability[f] = self.instability_df_single_feature(f)
        feature_instability = pd.DataFrame(feature_instability).T.sort_values('beta')
        return feature_instability
    
    def population_audit(self):
        self.init_counters()
        for arm in self.arms:
            f,b = arm
            D_tmp = self.impute_df(f,b)
            if self.is_categorical:
                D_tmp = pd.get_dummies(D_tmp)
            Y_imputed = self.clf.predict(D_tmp)
            self.F[arm] = accuracy_score(self.Y, Y_imputed, normalize=False)
            self.S[arm] = len(Y_imputed) - self.F[arm]
        self.quality_control()
        return
        
    def sample_posterios(self):
        rg = Generator(PCG64())
        theta = rg.beta(self.prior[0]+self.S, self.prior[1]+self.F) * self.scales
        #theta = np.random.beta(self.prior[0]+self.S, self.prior[1]+self.F) * self.scales
        return theta.idxmin()
    
    def sample_CDFs(self):
        P = self.CDFs / self.CDFs.sum()
        return np.random.choice(P.index, p=P.values)
    
    def sample_posterios_TT(self):
        rg = Generator(PCG64())
        theta = rg.beta(self.prior[0]+self.S, self.prior[1]+self.F) * self.scales
        return theta.nsmallest(2).index[0:2]
        
    def query_arm(self, arm):
        if self.is_scaled:
            self.query_scaled(arm)
        else:
            self.query(arm)
        return

    def query(self, arm):
        f,b = arm
        #pick a random data point
        #x = self.D.sample() This is slow
        #ID = x.index[0]
        ID = np.random.choice(self.D.index)
        
        if self.D.loc[ID][f] == b:
            self.F[arm] += 1
            return
        else:
            #impute
            x_imputed = self.impute(ID, f, b)
            #query the system
            Y_imputed = self.clf.predict(x_imputed)
            if Y_imputed[0] == self.Y[ID]:
                self.F[arm] += 1
            else:
                self.S[arm] += 1
        return
    
    def query_scaled(self, arm):
        f,b = arm
        #pick a random data point using data subsets
        ID = np.random.choice(self.subset_indices[arm])
        
        if self.D.loc[ID][f] == b:
            sys.exit('error: the value of feature %s should not be %s'%(f,b))
        else:
            #impute
            x_imputed = self.impute(ID, f, b)
            #query the system 
            Y_imputed = self.clf.predict(x_imputed)
            if Y_imputed[0] == self.Y[ID]:
                self.F[arm] += 1
            else:
                self.S[arm] += 1
        return

    def Uniform_measurement(self, T, confidence, scaled=False):
        self.init_counters(scaled)
        arms = self.arms.tolist()
        while T>0:
            random.shuffle(arms)
            for arm in arms:
                self.query_arm(arm)
                T -=1
                if T==0:
                    break
        self.quality_control()
        return self.DM_FWE_LB(confidence)
    
    def Uniform_decision(self, DM_level, confidence, max_query, scaled=False):
        self.init_counters(scaled)
        self.max_query = max_query
        CDFs = beta.cdf(DM_level, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)  
        arms = self.arms.tolist()
        while not self.DM_decision(confidence)[0]:
            random.shuffle(arms)
            for arm in arms:
                #query arm
                self.query_arm(arm)
                #update cdfs
                self.CDFs[arm] = beta.cdf(DM_level, self.prior[0]+self.S[arm], self.prior[1]+self.F[arm], scale=self.scales[arm])
                if self.DM_decision(confidence)[0]:
                    break
        self.quality_control()
        return self.DM_decision(confidence)
    
    def TS_measurement(self, T, confidence, scaled=False):
        self.init_counters(scaled)
        for t in range(T):
            #select arm
            selected_arm = self.sample_posterios()
            #query arm
            self.query_arm(selected_arm)
        self.quality_control()
        return self.DM_FWE_LB(confidence)
    
    def TS_decision(self, DM_level, confidence, max_query, scaled=False):
        self.init_counters(scaled)
        self.max_query = max_query
        CDFs = beta.cdf(DM_level, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)         
        while not self.DM_decision(confidence)[0]:
            #select arm
            selected_arm = self.sample_posterios()
            #query arm
            self.query_arm(selected_arm)
            #update cdfs
            self.CDFs[selected_arm] = beta.cdf(DM_level,
                                               self.prior[0]+self.S[selected_arm],
                                               self.prior[1]+self.F[selected_arm],
                                               scale = self.scales[selected_arm])
        self.quality_control()
        return self.DM_decision(confidence)
    
    def TTTS_measurement(self, T, confidence, scaled=False, TTP=0.5, limit=500):
        self.init_counters(scaled)
        for t in range(T):
            #select arm
            selected_arm = self.sample_posterios()
            #choose between selected arm and the challenger
            B = np.random.binomial(1, TTP, size=1)
            if B==0:
                challenger = self.sample_posterios()
                counter = 0
                while challenger == selected_arm:
                    challenger = self.sample_posterios()
                    counter += 1
                    if counter > limit:
                        print('Finding a challenger takes too long. Switched to approx method.')
                        arm1, arm2 = self.sample_posterios_TT()
                        challenger = arm2 if selected_arm == arm1 else arm1
                        if challenger == selected_arm:
                            print('Could not find a challenger.')
                            raise SystemExit()
                selected_arm = challenger
            #query arm
            self.query_arm(selected_arm)
        self.quality_control()
        return self.DM_FWE_LB(confidence)
        
    def TTTS_decision(self, DM_level, confidence, max_query, scaled=False, TTP=0.5, limit=500):
        self.init_counters(scaled)
        self.max_query = max_query
        CDFs = beta.cdf(DM_level, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)       
        while not self.DM_decision(confidence)[0]:
            #select arm
            selected_arm = self.sample_posterios()
            #choose between selected arm and the challenger
            B = np.random.binomial(1, TTP, size=1)
            if B==0:
                challenger = self.sample_posterios()
                counter = 0
                while challenger == selected_arm:
                    challenger = self.sample_posterios()
                    counter += 1
                    if counter > limit:
                        print('Finding a challenger takes too long. Switched to approx method.')
                        arm1, arm2 = self.sample_posterios_TT()
                        challenger = arm2 if selected_arm == arm1 else arm1
                        if challenger == selected_arm:
                            print('Could not find a challenger.')
                            raise SystemExit()
                selected_arm = challenger
            #query arm
            self.query_arm(selected_arm)
            #update cdfs
            self.CDFs[selected_arm] = beta.cdf(DM_level,
                                               self.prior[0]+self.S[selected_arm],
                                               self.prior[1]+self.F[selected_arm],
                                               scale = self.scales[selected_arm])
        self.quality_control()
        return self.DM_decision(confidence)
        
    def Greedy_measurement(self, T, confidence, scaled=False):
        self.init_counters(scaled)
        beta_max = self.DM_FWE_LB(confidence)
        CDFs = beta.cdf(beta_max, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)         
        for t in range(T):
            #select arm
            selected_arm = self.CDFs.idxmax()
            #query arm
            self.query_arm(selected_arm)
            #update beta
            beta_max = self.DM_FWE_LB(confidence)
            #update cdfs
            CDFs = beta.cdf(beta_max, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
            self.CDFs = pd.Series(CDFs, index=self.arms)         
        self.quality_control()
        return self.DM_FWE_LB(confidence)
    
    def Greedy_decision(self, DM_level, confidence, max_query, scaled=False):
        self.init_counters(scaled)
        self.max_query = max_query
        CDFs = beta.cdf(DM_level, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)         
        while not self.DM_decision(confidence)[0]:
            #select arm
            selected_arm = self.CDFs.idxmax()
            #query arm
            self.query_arm(selected_arm)
            #update cdfs
            self.CDFs[selected_arm] = beta.cdf(DM_level,
                                               self.prior[0]+self.S[selected_arm],
                                               self.prior[1]+self.F[selected_arm],
                                               scale = self.scales[selected_arm])
        self.quality_control()
        return self.DM_decision(confidence)
    
    def PM_measurement(self, T, confidence, scaled=False):
        self.init_counters(scaled)
        beta_max = self.DM_FWE_LB(confidence)
        CDFs = beta.cdf(beta_max, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)         
        for t in range(T):
            #select arm
            selected_arm = self.sample_CDFs()
            #query arm
            self.query_arm(selected_arm)
            #update beta
            beta_max = self.DM_FWE_LB(confidence)
            #update cdfs
            CDFs = beta.cdf(beta_max, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
            self.CDFs = pd.Series(CDFs, index=self.arms)         
        self.quality_control()
        return self.DM_FWE_LB(confidence)
    
    def PM_decision(self, DM_level, confidence, max_query, scaled=False):
        self.init_counters(scaled)
        self.max_query = max_query
        CDFs = beta.cdf(DM_level, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        self.CDFs = pd.Series(CDFs, index=self.arms)       
        while not self.DM_decision(confidence)[0]:
            #select arm
            selected_arm = self.sample_CDFs()
            #query arm
            self.query_arm(selected_arm)
            #update cdfs
            self.CDFs[selected_arm] = beta.cdf(DM_level,
                                               self.prior[0]+self.S[selected_arm],
                                               self.prior[1]+self.F[selected_arm],
                                               scale = self.scales[selected_arm])
        self.quality_control()
        return self.DM_decision(confidence)
    
    def query_all_arms(self, min_visit=1):
        #query each arm at least min_visit times
        #add this function at the begening of other audit functions to avoid not checking some arms at all.
        for i in range(min_visit):
            for arm in self.arms:
                self.query_arm(arm)
        return
    
    def lower_bounds(self, error):
        bounds = pd.Series(index = self.arms)
        for arm in self.arms:
            bounds[arm] = beta.ppf(error, self.prior[0]+self.S[arm], self.prior[1]+self.F[arm], scale=self.scales[arm])
        return bounds
    
    def CDF_sum(self, threshold):
        CDF = beta.cdf(threshold, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        return CDF.sum()
    
    def CDF_max(self, threshold):
        CDF = beta.cdf(threshold, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        return CDF.max()
    
    def CDF_mean(self, threshold):
        CDF = beta.cdf(threshold, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        return CDF.mean()
    
    def CCDF_product(self, threshold):
        CCDF = 1.0 - beta.cdf(threshold, self.prior[0]+self.S, self.prior[1]+self.F, scale=self.scales)
        return CCDF.prod()
        
    def DM_decision(self, confidence):
        if self.CDFs.sum() < (1-confidence):
            return(True,'YES',(self.F + self.S).sum())
        elif self.CDFs.max() > confidence:
            return(True,'NO', (self.F + self.S).sum())
        elif (self.F + self.S).sum() > self.max_query:
            return(True,'undecided', (self.F + self.S).sum())
        else:
            return(False,'unknown')

    def DM_FWE(self, confidence, binary_search_err= 0.0001):#assumes independent arms
        #Do a binary search to find beta
        L=0
        R=1
        while True:
            beta = (L+R)/2
            P = self.CCDF_product(beta)
            if abs(P - confidence) < binary_search_err:
                return beta
            elif P < confidence:
                R = beta
            else:
                L = beta
    
    def DM_FWE_LB(self, confidence, binary_search_err= 0.0001):#does not use independence
        alpha = 1 - confidence
        #Do a binary search to find beta
        L=0
        R=1
        while True:
            beta = (L+R)/2
            S = self.CDF_sum(beta)
            if abs(S - alpha) < binary_search_err:
                return beta
            elif S > alpha:
                R = beta
            else:
                L = beta
    
    def DM_FDR_ratio(self, rate, binary_search_err= 0.0001):
        #Do a binary search to find beta
        L=0
        R=1
        while True:
            beta = (L+R)/2
            FDR = self.CDF_mean(beta)
            if abs(FDR - rate) < binary_search_err:
                return beta
            elif FDR > rate:
                R = beta
            else:
                L = beta
    
    def DM_FDR_count(self, n_arms, binary_search_err= 0.0001):
        #Do a binary search to find beta
        L=0
        R=1
        while True:
            beta = (L+R)/2
            S = self.CDF_sum(beta)
            if abs(S - n_arms) < binary_search_err:
                return beta
            elif S > n_arms:
                R = beta
            else:
                L = beta
        
    def BestArm_mean(self,):#find the arm with minimum mean
        means = self.S /(self.S + self.F)
        return means.idxmin()
    
    def BestArm_MC(self, n_samples):#find the best arm using monte carlo
        wins = pd.Series(0, index = self.arms)
        for i in range(n_samples):
            rg = Generator(PCG64())
            theta = rg.beta(self.prior[0]+self.S, self.prior[1]+self.F) * self.scales
            wins[theta.idxmin()] +=1
        return wins.idxmax()
    
    def BestArm_Integrate(self, n_samples):#find the best arm by numerical inegration
        samples = np.linspace(0, 1, n_samples, endpoint=False)
        
        f = {}
        for arm in self.arms:
            f[arm] = beta.pdf(samples, self.prior[0]+self.S[arm], self.prior[1]+self.F[arm], scale=self.scales[arm])
        f = pd.DataFrame(f).T
        
        F = {}
        for arm in self.arms:
            F[arm] = 1.0 - beta.cdf(samples, self.prior[0]+self.S[arm], self.prior[1]+self.F[arm], scale=self.scales[arm])
        F = pd.DataFrame(F).T
        
        P = F.product()
        
        BAP= ((P*f)/F).sum(axis=1)
        return BAP.idxmax()
    
    def quality_control(self):
        self.query_count = self.S + self.F
        
        if 0 in self.query_count.values:
            print("Warning: some arm are not investigated at all.")
            
        if self.is_scaled:
            subset_sizes = self.subset_indices.apply(len)
            if (subset_sizes < self.query_count).any():
                print("Warning: more queries than the size of population for a single arm.")
        else:
            if self.query_count.max() > len(self.D):
                print("Warning: more queries than the size of population for a single arm.")
