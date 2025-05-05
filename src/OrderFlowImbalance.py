import pandas as pd
import numpy as np

class OFI:
    def __init__(self, csv_file_name):
        data = pd.read_csv(csv_file_name)
        self.levels = 10
        self.time_step = 200
        self.rows = len(data)
        self.price = data["price"].values
        self.OFI_coeffs = self.get_OFI_coeffs(data)
        self.best_OFI = self.get_best_OFI()
        self.all_multi_level_OFI = self.get_all_multi_level_OFI(data)
        self.first_PC = self.get_first_PC()
        self.all_integrated_OFI = self.get_all_integrated_OFI()
        
        return None
    
    def get_OFI_coeffs(self, data):
        OFI_coeffs = np.zeros((self.rows,2, self.levels),dtype=np.float64)
        for i in range(self.levels):
            P_m_b = data[OFI.get_col_name(i, bid=True, px=True)].values
            q_m_b = data[OFI.get_col_name(i, bid=True, px=False)].values
            conditions = [P_m_b[1:] - P_m_b[:-1] > 0,
                      P_m_b[1:] - P_m_b[:-1] == 0,
                     ]
            values = [q_m_b[1:],q_m_b[1:] - q_m_b[:-1]]
            OFI_coeffs[1:,0,i] = np.select(conditions, values, - q_m_b[1:])
            OFI_coeffs[0,0,i] = q_m_b[0]
        for i in range(self.levels):
            P_m_a = data[OFI.get_col_name(i, bid=False, px=True)].values
            q_m_a = data[OFI.get_col_name(i, bid=False, px=False)].values
            conditions = [P_m_a[1:] - P_m_a[:-1] > 0,
                      P_m_a[1:] - P_m_a[:-1] == 0,
                     ]
            values = [-q_m_a[1:],q_m_a[1:] - q_m_a[:-1]]
            OFI_coeffs[1:,1,i] = np.select(conditions, values, - q_m_a[1:])
            OFI_coeffs[0, 1, i] = -q_m_a[0]
        return OFI_coeffs
    
    def single_level_OFI(self, first_order, last_order, m):
        if last_order < first_order:
            raise ValueError("last order index should be smaller than first")
        return sum(self.OFI_coeffs[first_order:last_order+1, 0, m] - self.OFI_coeffs[first_order:last_order+1, 1, m])
    
    def get_best_OFI(self):
        best_OFI = np.zeros(self.rows-self.time_step, dtype=np.float64)
        for i in range(self.rows-self.time_step):
            best_OFI[i] = self.single_level_OFI(i,i+self.time_step, 0)
        return best_OFI
    
    def get_scaling_factor(self, data, M, first_order, last_order):
        prefactor = 1/(M+1) * (1/(2*(last_order-first_order+1)))
        summation = 0
        for i in range(M+1):
            summation += sum(data[OFI.get_col_name(i, bid = True, px = False)].values[first_order:last_order+1])
            summation += sum(data[OFI.get_col_name(i, bid = False, px = False)].values[first_order:last_order+1])
        return summation * prefactor
    
    def multi_level_OFI(self, data, first_order, last_order, M = 9):
        if last_order < first_order:
            raise ValueError("last order index should be smaller than first")
        OFI_vec = np.zeros(M+1, dtype=np.float64)
        scaling_factor = self.get_scaling_factor(data, M, first_order, last_order)
        for i in range(M+1):
            OFI_vec[i] = self.single_level_OFI(first_order, last_order, i)
        OFI_vec /= scaling_factor
        return OFI_vec
    
    def get_all_multi_level_OFI(self, data):
        all_multi_level_OFI = np.zeros((self.rows-self.time_step,self.levels),dtype=np.float64)
        for i in range(self.rows-self.time_step):
            all_multi_level_OFI[i,:] = self.multi_level_OFI(data, i, i+self.time_step, M=self.levels-1)
        return all_multi_level_OFI
    
    def get_first_PC(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        x = StandardScaler().fit_transform(self.all_multi_level_OFI.T)
        pca = PCA(n_components=self.levels)
        principal_components = pca.fit_transform(x)
        first_principal_component = principal_components[:,0]
        first_PC = first_principal_component/np.linalg.norm(first_principal_component,ord=1)
        return first_PC
    
    def get_all_integrated_OFI(self):
        all_integrated_OFI = np.zeros(self.rows-self.time_step,dtype=np.float64)
        for i in range(self.rows-self.time_step):
            all_integrated_OFI[i] = (self.first_PC.reshape((1,self.levels))@self.all_multi_level_OFI[i,:].reshape((self.levels,1)))[0,0]
        return all_integrated_OFI
        
    
    @staticmethod
    def get_col_name(m, bid=True, px=True):
        name = []
        if bid:
            name.append("bid_")
        else:
            name.append("ask_")
        if px:
            name.append("px_")
        else:
            name.append("sz_")
        name.append("0")
        name.append(str(m))
        return "".join(name)
    
if __name__ == "__main__":
    order_flow = OFI("../Data/first_25000_rows.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.set_ylabel("Best OFI")
    ax.plot(order_flow.best_OFI, label="best_OFI")
    fig.savefig("Figures/best_OFI", bbox_inches="tight")
    #plt.plot(order_flow.best_OFI, label="best_OFI")
    #plt.show()
    fig, ax = plt.subplots(1,1)
    ax.plot(order_flow.all_integrated_OFI, label="integrated_OFI")
    ax.set_ylabel("Integrated OFI")
    fig.savefig("Figures/all_integrated_OFI", bbox_inches="tight")
    #plt.plot(order_flow.all_integrated_OFI, label="integrated_OFI")
    #plt.show()
    print("The first principal component for the current data is :", order_flow.first_PC)
    fig, ax = plt.subplots(1,1)
    ax.plot(order_flow.price)
    ax.set_ylabel("AAPL Price")
    fig.savefig("Figures/price",bbox_inches="tight")
    #plt.plot(order_flow.price)
    #plt.show()
    #print(order_flow.OFI_coeffs)
        