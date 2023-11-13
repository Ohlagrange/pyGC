import os
import re
from natsort import ns, natsorted
from lxml import etree
import pandas as pd
import numpy as np
import xlwings as xw


'''
该程序用于辅助GT的计算及相关结果的后处理
平台环境：win10；GT v2016；python 3.8
主要功能如下：
## 修改gtm中的Case setup的设置
## 提交GT计算（GT模型关掉所有monitor窗口）
## 配置exp文件(仅限改文件名，具体配置项由于单位采用数字形式而不易用代码进行修改)
## 提取相关RLT结果输出到excel
## 提取相关曲线输出到excel
'''

#获得当前文件夹下所有整机模型名称，并自然排序
def models_name():
    filename=os.listdir('./')
    gtm=[]
    for i in filename:
        # .*表示任意个任意字符，$表示结尾
        ret=re.match(r'.*_DI.*.gtm$',i)
        if ret:
            gtm.append(i[:-4])
    #文件名自然排序
    gtm=natsorted(gtm,alg=ns.PATH)
    return gtm

def get_tree(file_name):
    parser=etree.XMLParser(strip_cdata=False)
    tree=etree.parse('%s' %file_name,parser)
    return tree

def modify_name(tree,tag,attribute,value):
    lst=tree.getroot()
    item=lst.findall('%s' %tag)
    for attr in item:
        attr.set('%s'%attribute,'%s' %value)
    return tree

def write_file(tree,file_name):
    tree.write('%s'%file_name,pretty_print=True,encoding=tree.docinfo.encoding,xml_declaration=True)

class gt_model():
    def __init__(self,file_name):
        self.file_name=file_name
    
    #获取总case数
    def total_case_num(self):
        file_name=self.file_name
        case_number=0
        parser=etree.XMLParser(strip_cdata=False)
        tree=etree.parse('%s.gtm' %file_name,parser)
        gtm=tree.getroot()
        lst=gtm.findall('cases/case')
        for case in lst:
            if case.get('on')=='0':
                continue
            else:case_number=case_number+1
        return case_number
        
    #根据para_name和case_No获取参数值
    def getPara(self,para_name,case_No):
        file_name=self.file_name
        all_para_value=[]
        parser=etree.XMLParser(strip_cdata=False)
        tree=etree.parse('%s.gtm' %file_name,parser)
        gtm=tree.getroot()
        lst=gtm.findall('cases/case')
        for case in lst:
            if case.get('on')=='0':
                continue
            else:
                item=case.findall('item')
                for para in item:
                    if para.get('parameter')==para_name:
                        all_para_value.append(para.get('text'))
        para_value=all_para_value[case_No-1]
        return para_value

    def getRLT_all(self):
        file_name=self.file_name
        df=pd.read_csv('%s_RLT.txt'%file_name,sep='\t',skiprows=1,header=None)
        index=df.iloc[:,0].values.tolist()
        df=df.iloc[:,1:]
        df.index=list(index)
        df.columns = range(df.shape[1])
        return df

    #根据case_No获取曲线结果,Plot或者SecondLaw
    def getXY(self,type,case_No):
        file_name=self.file_name
        with open('%s_%s.txt'%(file_name,type),'r') as f:
            i=-1
            data_case=[]
            for line in f.readlines():
                if '#' in line:
                    i=i+1
                    data_case.append([])
                elif line=='\n':
                    continue
                else:
                    line=line.replace('\n','')
                    data_case[i].append(line.split('\t'))           
        data_plot=pd.DataFrame(data_case[case_No-1])
        data_plot.dropna(axis=1,inplace=True)
        col=data_plot[:1].values.tolist()
        data_plot=data_plot[2:]
        data_plot.columns=list(col)
        data_plot.reset_index(drop=True, inplace=True)
        #name_plot=data_plot.loc[:,[long_name+'-X',long_name+'-Y']]
        #name_plot=name_plot.astype(float)
        return data_plot

    def run(self):
        file_name=self.file_name
        os.system('gtsuite -m:off %s.gtm' %file_name)

    def exportRLT(self):
        file_name=self.file_name
        tree=get_tree('init_RLT.exp')
        tree=modify_name(tree,'source_file','name','%s.gdx'%file_name)
        tree=modify_name(tree,'target_file','name','%s_RLT.txt'%file_name)
        write_file(tree,'%s_RLT_config.exp'%file_name)
        os.system('gtexport %s_RLT_config.exp' %file_name)    
    
    #需要事先做好初始文件init_Plot.exp
    def exportPlot(self):
        file_name=self.file_name
        tree=get_tree('init_Plot.exp')
        tree=modify_name(tree,'source_file','name','%s.gdx'%file_name)
        tree=modify_name(tree,'target_file','name','%s_Plot.txt'%file_name)
        write_file(tree,'%s_Plot_config.exp'%file_name)
        os.system('gtexport %s_Plot_config.exp' %file_name)

    #依赖于RLT.txt,每个case顺序输出
    #time:GLOBAL
    def exportSecondLaw(self):
        file_name=self.file_name
        total_case=self.total_case_num()
        tree=get_tree('init_SecondLaw.exp')
        tree=modify_name(tree,'source_file','name','%s.gdx'%file_name)
        tree=modify_name(tree,'target_file','name','%s_SecondLaw_single.txt'%file_name)
        f=open('%s_SecondLaw.txt'%file_name,'w+')
        f.close()
        for i in range(1,total_case+1):
            #注意exp文件中deg的设定
            tree=modify_name(tree,'export_info/case_info/export_case_number','case_number','%d'%i)
            end_value=self.getRLT_value('time:GLOBAL',i)
            RPM=self.getRLT_value('Engine Speed (cycle average); Part engine-1',i)
            increment_value=1/(RPM/60)/720 #0.5deg
            start_value=end_value-increment_value*720
            tree=modify_name(tree,'export_info/case_info/export_case_number','case_number','%d'%i)
            lst=tree.getroot()
            item=lst.findall('interpolation_options/interpolation_parameter')
            for tag in item:
                if tag.get('interpolation_param_type')=='2':
                    tag.set('end_value','%s'%end_value)
                    tag.set('increment_value','%s'%increment_value)
                    tag.set('start_value','%s'%start_value)
            write_file(tree,'%s_SL_config.exp'%file_name)
            os.system('gtexport %s_SL_config.exp' %file_name)
            file=open('%s_SecondLaw.txt'%file_name,'a+')
            for line in open('%s_SecondLaw_single.txt'%file_name,encoding='utf-8'):
                file.writelines(line)
            file.write('\n')
            file.close()

    def load_longname(self,filename):
        names=[]
        for line in open(filename):
            if line=='\n':
                continue
            else: 
                line=line.replace('\n','')
                names.append(line)
        return names

    #自定义需要的结果
    def getRLT_user(self,long_name):
        df=self.getRLT_all()
        df=df.loc[long_name]
        return df

    def getRLT_value(self,long_name,case_No):
        df=self.getRLT_all()
        value=df.loc[long_name,case_No]
        return value

def RLT2Excel_all(RLT_longname,ExcelFileName,SheetName):
    models=models_name()
    output=pd.DataFrame([])
    i=0
    for name in models:
        model=gt_model(name)
        # 第一次运行打开
        #model.exportRLT()
        longname=model.load_longname(RLT_longname)
        df=model.getRLT_user(longname)
        if i==0:
            output=pd.concat([output,df],axis=1)
            i=i+1
        else:
            output=pd.concat([output,df.iloc[:,1:]],axis=1)
    output.columns = range(output.shape[1])
    app=xw.App(visible=False,add_book=False)
    wb=app.books.open(ExcelFileName)
    sht=wb.sheets[SheetName]
    #excel第二行第三列，C2
    sht.range(1,2).value=output
    sht.range('B1').column_width=60
    sht[1,:].color=(146, 208, 80) #浅绿色
    wb.save()
    wb.close()
    app.quit()

def Plot2Excel_all():
    models=models_name()
    for name in models:
        model=gt_model(name)
        model.exportPlot()
        for i in range(1,model.total_case_num()+1):
            rpm=model.getRLT_value('Engine Speed (end of cycle); Part engine-1',i)
            app = xw.App(visible=False,add_book=True)
            wb=app.books.add()
            output = model.getXY('Plot',i)
            sht=wb.sheets[0]
            sht.range(1,1).value=output
            sht.range('A:A').clear()
            ##公式设置##
            k=0
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dHT_intake'
                sht.range(j,output.shape[1]+5+k).formula='=(E%d-F%d)*C%d*1/(%f/60)/720'%(j,j,j,rpm)
            sht.range(2+k,output.shape[1]+2).value='HT_intake(J)'
            sht.range(2+k,output.shape[1]+3).formula='=SUM(P:P)'
            k=k+1
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dHT_exhaust'
                sht.range(j,output.shape[1]+5+k).formula='=(H%d-I%d)*G%d*1/(%f/60)/720'%(j,j,j,rpm)
            sht.range(2+k,output.shape[1]+2).value='HT_exhaust(J)'
            sht.range(2+k,output.shape[1]+3).formula='=0'
            k=k+1
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dA_ambient'
                sht.range(j,output.shape[1]+5+k).formula='=(J%d*G%d-D%d*C%d)*1/(%f/60)/720'%(j,j,j,j,rpm)  
            sht.range(2+k,output.shape[1]+2).value='A_ambient(J)'
            sht.range(2+k,output.shape[1]+3).formula='=SUM(R:R)'  
            k=k+1         
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dIrre_Comp'
                sht.range(j,output.shape[1]+5+k).formula='=(D%d-E%d)*C%d*1/(%f/60)/720'%(j,j,j,rpm)  
            sht.range(2+k,output.shape[1]+2).value='Irre_Comp(J)'
            sht.range(2+k,output.shape[1]+3).formula='=SUM(S:S)+N8*1000*1/(%f/60)'%rpm 
            k=k+1 
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dIrre_Turb'
                sht.range(j,output.shape[1]+5+k).formula='=(I%d-J%d)*G%d*1/(%f/60)/720'%(j,j,j,rpm)  
            sht.range(2+k,output.shape[1]+2).value='Irre_Turb(J)'
            sht.range(2+k,output.shape[1]+3).formula='=SUM(T:T)-N9*1000*1/(%f/60)'%rpm 
            k=k+1            
            for j in range(2,723):
                sht.range(1,output.shape[1]+5+k).value='dHT_cyl-1'
                sht.range(j,output.shape[1]+5+k).formula='=(1-298.15/L%d)*K%d*1000*1/(%f/60)/720'%(j,j,rpm)            
            sht.range(2+k,output.shape[1]+2).value='HT_cyl(J)'
            sht.range(2+k,output.shape[1]+3).formula='=SUM(U:U)*6'
            k=k+1 
            sht.range(2+k,output.shape[1]+2).value='AveragePower_Comp(kW)'
            sht.range(2+k,output.shape[1]+3).value=model.getRLT_value('Average Power (Incl. Shaft if Modeled); Part compressor-1',i)
            k=k+1
            sht.range(2+k,output.shape[1]+2).value='AveragePower_Turb(kW)'
            sht.range(2+k,output.shape[1]+3).value=model.getRLT_value('Average Power (Incl. Shaft if Modeled); Part turbine-1',i)            
            k=k+1
            sht.range(2+k,output.shape[1]+2).value='Irre_TC_friction(J)'
            sht.range(2+k,output.shape[1]+3).formula='=(N9-N8)*1000*1/(%f/60)'%rpm            
            wb.save('plot_data/Case%d_Plot.xlsx'%model.getRLT_value('Case',i))
            wb.close()
            app.quit()       

def Second2Excel_all():
    models=models_name()
    for name in models:
        model=gt_model(name)
        model.exportSecondLaw()
        for i in range(1,model.total_case_num()+1):
            output=pd.DataFrame([])
            rpm=model.getRLT_value('Engine Speed (end of cycle); Part engine-1',i)
            LHV=model.getRLT_value('Lower Heating Value of Injected Fuel; Part inject',i)
            txtdata = model.getXY('SecondLaw',i)
            output=txtdata.loc[0:720,['Pressure:cyl-1(bar)-X',
                                         'CV_Avail:cyl-1(J)-Y',
                                         'Burn Rate:cyl-1(mg/deg)-Y',
                                         'Pressure:cyl-1(bar)-Y',
                                         'Volume:cyl-1(L)-Y',
                                         'HTRate:cyl-1(kW)-Y',
                                         'Temperature:cyl-1(K)-Y',
                                         'AvailFlux:exhaust-1(J/s)-Y',
                                         'ChmAvailFlux:exhaust-1(J/s)-Y',
                                         'AvailFlux:intake-1(J/s)-Y',
                                         'ChmAvailFlux:intake-1(J/s)-Y'
                                         ]]
            app = xw.App(visible=False,add_book=True)
            wb=app.books.add()
            sht=wb.sheets[0]
            sht.range(1,1).value=output
            sht.range('A:A').clear()
            sht.range('B1').value='CrankAngle(deg)'
            ##公式设置##
            k=0
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Agas(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=(C%d-C%d)/(B%d-B%d)/1000'%(j+1,j,j+1,j)
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Afuel(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=D%d*%f*1.066/1000'%(j,LHV)
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Awork(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=(E%d+E%d-2)/2*(F%d-F%d)/10/(B%d-B%d)'%(j+1,j,j+1,j,j+1,j)
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='AHT(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=(1-298.15/H%d)*G%d/(%f/60)/360'%(j,j,rpm)
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Aexh(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=(I%d-J%d-(K%d-L%d))/1000/(%f/60)/360'%(j,j,j,j,rpm)
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Irre_combustion(kJ/deg)'
                sht.range(j,output.shape[1]+3+k).formula='=O%d-N%d-P%d-Q%d-R%d'%(j,j,j,j,j) 
            k=k+1
            for j in range(2,722):
                sht.range(1,output.shape[1]+3+k).value='Irre(kJ)'
                sht.range(j,output.shape[1]+3+k).formula='=SUM(S$2:$S%d)*(B%d-B%d)'%(j,j+1,j)             
            wb.save('plot_data/Case%d_SL.xlsx'%model.getRLT_value('Case',i))
            wb.close()
            app.quit() 
    return 0

def Balance2Excel_all(ExcelFileName,SheetName):
    excelfiles=os.listdir('plot_data')
    num=len(excelfiles)
    output=[]
    #case_start
    for i in range(1,5):
        results=[]
        app=xw.App(visible=False,add_book=False)
        wb=app.books.open('plot_data/Case%d_Plot.xlsx'%i)
        sht=wb.sheets[0]
        results.append(sht.range('N7').value)
        results.append(sht.range('N2').value)
        results.append(sht.range('N3').value)
        results.append(sht.range('N4').value)
        results.append(sht.range('N5').value)
        results.append(sht.range('N6').value)
        results.append(sht.range('N10').value)
        wb.close()
        app.quit()         
        app=xw.App(visible=False,add_book=False)
        wb=app.books.open('plot_data/Case%d_SL.xlsx'%i)
        sht=wb.sheets[0]        
        results.append(6000*sht.range('T402').value)
        wb.close()
        app.quit()
        output.append(results)
    app=xw.App(visible=False,add_book=False)
    wb=app.books.open(ExcelFileName)
    sht=wb.sheets[SheetName]
    output=np.array(output)
    sht.range('D2').value=output.T
    wb.save()
    wb.close()
    app.quit() 

if __name__=='__main__':
    #RLT2Excel_all('user_RLT_longname_1.txt','tmp.xlsx','Sheet1')
    #RLT2Excel_all('user_RLT_longname_thermal.txt','Results.xlsx','Sheet1')
    #Plot2Excel_all()
    #Second2Excel_all()
    Balance2Excel_all('Second.xlsx','Sheet1')


'''
            name_plot=txtdata.loc[0:720,['Pressure:cyl-1(bar)-X',
                                         'CV_Avail:cyl-1(J)-Y',
                                         'Burn Rate:cyl-1(mg/deg)-Y',
                                         'Pressure:cyl-1(bar)-Y',
                                         'Volume:cyl-1(L)-Y',
                                         'HTRate:cyl-1(kW)-Y',
                                         'Temperature:cyl-1(K)-Y',
                                         'AvailFlux:exhaust-1(J/s)-Y',
                                         'ChmAvailFlux:exhaust-1(J/s)-Y',
                                         'AvailFlux:intake-1(J/s)-Y',
                                         'ChmAvailFlux:intake-1(J/s)-Y'
                                         ]]

            CrankAngle=txtdata.loc[0:720,['Pressure:cyl-1(bar)-X']]
            CV_A=txtdata.loc[0:720,['CV_Avail:cyl-1(J)-Y']]
            BurnRate=txtdata.loc[0:720,['Burn Rate:cyl-1(mg/deg)-Y']]
            P_cyl=txtdata.loc[0:720,['Pressure:cyl-1(bar)-Y']]
            V_cyl=txtdata.loc[0:720,['Volume:cyl-1(L)-Y',]]
            HTRate_cyl=txtdata.loc[0:720,['HTRate:cyl-1(kW)-Y']]
            T_cyl=txtdata.loc[0:720,['Temperature:cyl-1(K)-Y']]
            AvailFlux_exh=txtdata.loc[0:720,['AvailFlux:exhaust-1(J/s)-Y']]
            ChmAvailFlux_exh=txtdata.loc[0:720,['ChmAvailFlux:exhaust-1(J/s)-Y']]
            AvailFlux_int=txtdata.loc[0:720,['AvailFlux:intake-1(J/s)-Y']]
            ChmAvailFlux_int=txtdata.loc[0:720,['ChmAvailFlux:intake-1(J/s)-Y']]
'''