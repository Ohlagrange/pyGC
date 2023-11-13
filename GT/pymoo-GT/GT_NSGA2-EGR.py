from lxml import etree
import pandas as pd
import numpy as np
from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
import os
from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
from pymoo.optimize import minimize
import multiprocessing
from pymoo.core.problem import StarmapParallelization
import subprocess
import shutil
import csv
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS

class MyProblem(ElementwiseProblem):

    def __init__(self,**kwargs):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_constr=6,
                         #喘振裕度>10%,Pmax<170bar,deltaP<45bar,Cycle<110,NOx<3.4(需要更改RLT)
                         xl=np.array([120, 250, -3, 30, 30]),
                         xu=np.array([155, 275, 3, 45, 45]),
                         **kwargs)
        
    def _evaluate(self, x, out, *args, **kwargs):

        #update_dijet_param(x)
        # os.system(r'0_run.bat')
        #subprocess.call(r'0_run.bat', shell=True)
        #RLT_list=['Burn Rate RMS Error (Meas vs Pred); Part cyl-1']
        #RLT_value = get_RLT_value("340_DIJet_combustion_cali", RLT_list)

        # 使用folder_counter获取文件夹名称
        subdir = os.path.join("Double_Load50_EGR", 'EVO_%s_EVC_%s_SOI_%s_EGR_%s_CB_%s'%(x[0], x[1], x[2], x[3], x[4]))

        os.makedirs(subdir, exist_ok=True)

        # 复制input文件夹的内容到子文件夹
        for file in os.listdir('input'):
            shutil.copyfile(os.path.join("input", file), os.path.join(subdir, file))

        # 修改update_dijet_param和subprocess.call以使用子文件夹
        #x = np.append(x, [0.84, 0.99])
        update_dijet_param(x, os.path.join(subdir, 'inputs.param'))
        #os.system(r'0_run.bat')
        subprocess.call(os.path.join(subdir, '0_run.bat'), shell=True)

        RLT_value = get_RLT_value(os.path.join(subdir, "RLT.txt"))
        print(RLT_value)

        f1 = RLT_value[0]
        f2 = RLT_value[1]

        # 需要满足 g < 0
        g1 = 10 - RLT_value[2] 
        g2 = RLT_value[3] - 135
        # 100%负荷RLT_value[3]/RLT_value[4] - 1.25
        g3 =  RLT_value[3] - RLT_value[4] - 45
        g4 = RLT_value[5] - 110 + 0.5

        g5 = RLT_value[1] -3.4
        g6 = 3 - RLT_value[1]

        out["F"] = [f1, f2]
        out["G"] = [g1, g2, g3, g4, g5, g6]


def update_dijet_param(x, file):
    # 确认x是一个numpy数组，并且有4个元素
    if not isinstance(x, np.ndarray) or len(x) != 5:
        raise ValueError("x must be a numpy array with 5 elements.")
    
    # 定义要更改的参数名称
    param_names = ["EVO", "EVC", "SOI", 'EGR_rate', 'BP-Rate']
    
    # 生成新的文件内容
    content = []
    for name, value in zip(param_names, x):
        content.append(f"{name}> {value:.4f}<\n")
    
    # 将新的内容写入文件
    with open(file, "w") as file:
        file.writelines(content)


def get_RLT_value(filepath):
    # 如果文件不存在，直接返回[10, 10, 10]
    if not os.path.exists(filepath):
        return [1000, 1000, 10000, 1000, 1000, 1000]
    
    try:
        with open(filepath, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            data = list(reader)
            
            # 检查是否有第三行
            if len(data) < 3:
                return [1000, 1000, 10000, 1000, 1000, 1000]
            
            # 从第三行开始
            BSFC = data[2][2]
            NOx = data[3][2]
            SurgeMargin = data[4][2]
            Pmax = data[5][2]
            Pcomp = data[6][2]
            Cycle = data[7][2]            
            
            # 转换为float类型，如果转换失败，设置该项值为10
            BSFC_value = float(BSFC) if BSFC.replace('.', '', 1).isdigit() else 1000 
            NOx_value = float(NOx) if NOx.replace('.', '', 1).isdigit() else 1000 
            SurgeMargin_value = float(SurgeMargin) if SurgeMargin.replace('.', '', 1).isdigit() else 1000 
            Pmax_value = float(Pmax) if Pmax.replace('.', '', 1).isdigit() else 1000 
            Pcomp_value = float(Pcomp) if Pcomp.replace('.', '', 1).isdigit() else 1000 
            Cycle_value = float(Cycle) if Cycle.replace('.', '', 1).isdigit() else 1000 

            return [BSFC_value, NOx_value, SurgeMargin_value, Pmax_value, Pcomp_value, Cycle_value]
    except Exception as e:  # 如果读取中出现任何问题，都返回[10, 10, 10, 10]
        print(f"Error reading the file: {e}")
        return [1000, 1000, 10000, 1000, 1000, 1000]


if __name__=='__main__':

    n_proccess = 28
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    #termination = get_termination("n_max_evals", 300)
    termination = DefaultSingleObjectiveTermination(n_max_evals=300)
    #termination = get_termination("n_gen", 20)

    algorithm_3 = PSO(pop_size=24,
                      sampling=FloatRandomSampling(),
                      eliminate_duplicates=True)

    algorithm_4 = NSGA2(pop_size=24,
                        sampling=LHS(),
                        eliminate_duplicates=True)
    
    # Define reference points [f1, f2]
    ref_points = np.array([[175, 12], [170, 14]])

    # Get Algorithm
    algorithm_5 = RNSGA2(
        ref_points=ref_points,
        pop_size=24,
        epsilon=0.01,
        normalization='front',
        extreme_points_as_reference_points=False,
        weights=np.array([0.5, 0.5]))
        


    #problem = MyProblem()
    problem = MyProblem(elementwise_runner=runner)
    res = minimize(problem,
               algorithm_4,
               termination,
               seed=1,
               verbose=False)
    np.savetxt('X.txt',res.X)
    np.savetxt('Y.txt',res.F)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    print("--- %s seconds ---" %res.exec_time)    