from xlrd import open_workbook, xldate_as_tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from glob import glob
from collections import defaultdict
import os

############################################################
NUM_AGE_GROUPS = 26
#Cohort characteristics
AGE, RISK, CARE = range(3)
CHARACTERISTICS = (AGE, RISK, CARE)
#Risk states
PERI_RISK, MSM_HIGH_RISK, MSM_LOW_RISK, FSW_RISK, IDU_RISK, LOW_RISK = range(6)
RISK_STATES = (PERI_RISK, MSM_HIGH_RISK, MSM_LOW_RISK, FSW_RISK, IDU_RISK, LOW_RISK)
RISK_STRS = ("Peri", "MSM_HIGH", "MSM_LOW", "FSW", "IDU", "LOW")
#Care states (Never in care, in care, ltfu)
NEVER_CARE, IN_CARE, LTFU_CARE = range(3)
CARE_STATES = (NEVER_CARE, IN_CARE, LTFU_CARE)
CARE_STRS = ("Never in care", "In Care","LTFU")
IU_IP, PP = range(2)

#Wrapper for defaultdict which allows pretty printing
class PrettyDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)
    def __repr__(self):
        return dict(self).__repr__()
    
#factory for arbitray levels of nested dictionaries
def multi_dict():
    return PrettyDict(multi_dict)

############################################################
class Inputs(object):
    """Class for model inputs"""
    def __init__(self):
        """Cohort Inputs"""
        #Number of years to run model
        self.runtime = None
        #Year of model start
        self.startyear = None
        #Number of births by year
        self.num_births = []

        """Generated Cohort inputs"""
        #Number of prevalent children by age, risk, and care
        self.num_prevalent = multi_dict()
        #Overall Population of each risk category (excluding perinatal) by age, year, and risk
        self.num_risk = multi_dict()

        """Incidence Inputs"""
        #Incidence rate by age, year, and risk
        self.inc_rate = multi_dict()
        
        """MTCT Inputs"""
        #PMTCT coverage by year
        self.pmtct = []
        #Prevalence among pregnant mothers by year
        self.prev_mothers = []
        #Prob breastfeed by year
        self.bf = []
        #MTCT rate by year
        self.mtct_with_pmtct_iuip = []
        self.mtct_with_pmtct_pp = []
        self.mtct_without_pmtct_iuip = []
        self.mtct_without_pmtct_pp = []

        """Follow up inputs"""
        #Prob of accessing diagnosis by age, year, and risk
        self.diagnosis = multi_dict()
        #prob of linkage by age, year, and risk
        self.linkage = multi_dict()
        #prob of ltfu/cycle by age, year, and risk
        self.ltfu = multi_dict()
        #prob of rtc/cycle by age, year, and risk
        self.rtc = multi_dict()

        """Survival inputs"""
        #Prob survive by age, risk, care, and year
        self.survival = multi_dict()

############################################################
class Outputs(object):
    """Class for model outputs"""
    def initialize_outputs(self, runtime):
        """
        Initialize numpy array for each output.
        Must be called after inputs are read so we know runtime
        """
        #Number of people alive before deaths are calcluated by age, risk, care, and year
        self.num_alive = np.zeros((NUM_AGE_GROUPS, len(RISK_STATES), len(CARE_STATES), runtime))
        #Number of deaths by age, risk, care, and year
        self.num_deaths = np.zeros((NUM_AGE_GROUPS, len(RISK_STATES), len(CARE_STATES), runtime))
        #Number LtFU by age, risk, year
        self.num_ltfu = np.zeros((NUM_AGE_GROUPS, len(RISK_STATES), runtime))
        #Number RTC by age, risk, year
        self.num_rtc = np.zeros((NUM_AGE_GROUPS, len(RISK_STATES), runtime))
        #Number incident infections by age, risk, year
        self.num_inf = np.zeros((NUM_AGE_GROUPS, len(RISK_STATES), runtime))
############################################################
class Population(object):
    """
    Container class for the numbers of people
    in each subgroup of the population
    """
    def __init__(self):
        #cohort will be a numpy array with n dimensions
        #n is the number of cohort characteristics
        #The characteristics are AGE, GENDER, and CARE STATE
        self.cohort = np.zeros((NUM_AGE_GROUPS,len(RISK_STATES),len(CARE_STATES)))
    def __setitem__(self, key, value):
        self.cohort.__setitem__(key, value)
    def __getitem__(self, key):
        return self.cohort.__getitem__(key)
    def __delitem__(self, key):
        self.cohort.__delitem__(key)
    def __str__(self):
        #prints population in readable format
        print_str = ""

        for k in CARE_STATES:
            print_str += "\n{}".format(CARE_STRS[k])
            print_str += "\nAge\t"+"\t".join(RISK_STRS)
            for i in range(NUM_AGE_GROUPS):
                print_str += "\n{}".format(i)
                for j in RISK_STATES:
                    print_str += "\t{}".format(self.cohort[i][j][k])
            print_str += "\n"
        return print_str
    def age_pop(self):
        #Ages the population one year while discarding the last age bucket
        old_cohort = np.copy(self.cohort)
        self.cohort[0] = 0
        for i in range(NUM_AGE_GROUPS - 1):
            self.cohort[i+1] = old_cohort[i]
        
############################################################
class Sim(object):
    """Class for simulation.  Stores all inputs/outputs."""
    def __init__(self):
        self.inputs = Inputs()
        self.outputs = Outputs()
    def init_run(self):
        """Initializes run parameters"""
        #Set time t = -1, the first cycle will make the time t=0
        self.time_step = -1
        
        #create blank population
        self.population = Population()
        #Seed initial cohort
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                for k in CARE_STATES:
                    self.population[i][j][k] = self.inputs.num_prevalent[i][j][k]
    def step(self):
        """
        run one cycle(year) of the model
        The cyle is LTFU, perinatal infections, non perinatal infections, deaths, aging up
        """
        #increment time step
        self.time_step+=1
        t = self.time_step

        #LTFU and RTC
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                num_ltfu = self.population[i][j][IN_CARE]*self.inputs.ltfu[i][t][j]
                num_rtc = self.population[i][j][LTFU_CARE]*self.inputs.rtc[i][t][j]
                net_ltfu = num_ltfu - num_rtc
                self.population[i][j][LTFU_CARE] += net_ltfu
                self.population[i][j][IN_CARE] -= net_ltfu
                self.outputs.num_ltfu[i][j][t] = num_ltfu
                self.outputs.num_rtc[i][j][t] = num_rtc

        #perinatal infections
        num_born = self.inputs.num_births[t]
        pmtct = self.inputs.pmtct[t]
        #number IU/IP
        num_iuip = num_born*self.inputs.prev_mothers[t]*\
                (pmtct*self.inputs.mtct_with_pmtct_iuip[t]+\
                (1-pmtct)*self.inputs.mtct_without_pmtct_iuip[t])
        #Number born with pp infection
        num_pp = num_born*self.inputs.prev_mothers[t]*self.inputs.bf[t]*\
                (pmtct*(1-self.inputs.mtct_with_pmtct_iuip[t])*self.inputs.mtct_with_pmtct_pp[t]+\
                (1-pmtct)*(1-self.inputs.mtct_without_pmtct_iuip[t])*self.inputs.mtct_without_pmtct_pp[t])

        num_inf = num_iuip+num_pp

        #perinatal infections that go into care
        self.population[0][PERI_RISK][NEVER_CARE] += num_inf
        self.outputs.num_inf[0][PERI_RISK][t] = num_inf

        #new infections for other risk groups
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                if j == PERI_RISK:
                    num_inf = 0
                else:
                    num_prev = sum(self.population[i][j])
                    #Num Infected is the incidence rate multiplied by num eligible
                    num_inf = (self.inputs.num_risk[i][t][j]-num_prev)*self.inputs.inc_rate[i][t][j]
                    self.population[i][j][NEVER_CARE] += num_inf
                if not(i==0 and j==PERI_RISK):
                    self.outputs.num_inf[i][j][t] = num_inf

        #Transition those in never care group to in care group
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                num_link = self.population[i][j][NEVER_CARE]*self.inputs.diagnosis[i][t][j]*self.inputs.linkage[i][t][j]
                self.population[i][j][IN_CARE]+=num_link
                self.population[i][j][NEVER_CARE]-=num_link

        #write output for num_alive
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                for k in CARE_STATES:
                    self.outputs.num_alive[i][j][k][t] = self.population[i][j][k]
        
        #Deaths
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                for k in CARE_STATES:
                    num_deaths = self.population[i][j][k]* (1.0-self.inputs.survival[i][j][k][t])
                    self.population[i][j][k] -= num_deaths
                    self.outputs.num_deaths[i][j][k][t] = num_deaths

        #Age up
        self.population.age_pop()

    def run(self):
        """runs the model for the number of time steps specified in runtime"""
        for i in range(self.inputs.runtime):
            self.step()
    def write_outputs(self, filepath):
        with open(filepath, 'w') as fOut:
            #Composite Outputs                       
            for j in RISK_STATES:
                fOut.write("\t{}".format(RISK_STRS[j]))
            fOut.write("\tTotal\tDeaths\tLTFU\tRTC\tNew Infections")
            for t in range(self.inputs.runtime):
                num_alive = 0
                num_death = 0
                num_ltfu = 0
                num_rtc = 0
                num_inf = 0
                fOut.write("\nYear {}".format(self.inputs.startyear + t))
                for j in RISK_STATES:
                    num_risk = 0
                    for i in range(NUM_AGE_GROUPS):
                        num_ltfu += self.outputs.num_ltfu[i][j][t]
                        num_rtc += self.outputs.num_rtc[i][j][t]
                        num_inf += self.outputs.num_inf[i][j][t]
                        for k in CARE_STATES:
                            num_risk += self.outputs.num_alive[i][j][k][t]
                            num_death += self.outputs.num_deaths[i][j][k][t]

                    fOut.write("\t{}".format(num_risk))
                    num_alive+=num_risk
                fOut.write("\t{}".format(num_alive))
                fOut.write("\t{}".format(num_death))
                fOut.write("\t{}".format(num_ltfu))
                fOut.write("\t{}".format(num_rtc))
                fOut.write("\t{}".format(num_inf))
                
            fOut.write("\n")
            #Number Alive
            for k in CARE_STATES:
                fOut.write("\n")
                fOut.write("Number Alive")
                fOut.write("\n"+CARE_STRS[k])
                fOut.write("\n\t")
                for t in range(self.inputs.runtime):
                    if t!= 0:
                        fOut.write("\t"*(len(RISK_STATES)+1))
                    fOut.write("Year {}".format(self.inputs.startyear + t))
                fOut.write("\nAge")
                for t in range(self.inputs.runtime):
                    if t != 0:
                        fOut.write("\t")
                    for j in RISK_STATES:
                        fOut.write("\t"+RISK_STRS[j])
                for i in range(NUM_AGE_GROUPS):
                    fOut.write("\n{}".format(i))
                    for t in range(self.inputs.runtime):
                        for j in RISK_STATES:
                            fOut.write("\t{}".format(self.outputs.num_alive[i][j][k][t]))
                        fOut.write("\t")
                fOut.write("\n")

            #Number Deaths
            for k in CARE_STATES:
                fOut.write("\n")
                fOut.write("Number Deaths")
                fOut.write("\n"+CARE_STRS[k])
                fOut.write("\n\t")
                for t in range(self.inputs.runtime):
                    if t!= 0:
                        fOut.write("\t"*(len(RISK_STATES)+1))
                    fOut.write("Year {}".format(self.inputs.startyear + t))
                fOut.write("\nAge")
                for t in range(self.inputs.runtime):
                    if t != 0:
                        fOut.write("\t")
                    for j in RISK_STATES:
                        fOut.write("\t"+RISK_STRS[j])
                for i in range(NUM_AGE_GROUPS):
                    fOut.write("\n{}".format(i))
                    for t in range(self.inputs.runtime):
                        for j in RISK_STATES:
                            fOut.write("\t{}".format(self.outputs.num_deaths[i][j][k][t]))
                        fOut.write("\t")
                fOut.write("\n")

            #Number LTFU
            fOut.write("\n")
            fOut.write("Number Newly LTFU")
            fOut.write("\n\t")
            for t in range(self.inputs.runtime):
                if t!= 0:
                    fOut.write("\t"*(len(RISK_STATES)+1))
                fOut.write("Year {}".format(self.inputs.startyear + t))
            fOut.write("\nAge")
            for t in range(self.inputs.runtime):
                if t != 0:
                    fOut.write("\t")
                for j in RISK_STATES:
                    fOut.write("\t"+RISK_STRS[j])
            for i in range(NUM_AGE_GROUPS):
                fOut.write("\n{}".format(i))
                for t in range(self.inputs.runtime):
                    for j in RISK_STATES:
                        fOut.write("\t{}".format(self.outputs.num_ltfu[i][j][t]))
                    fOut.write("\t")
            fOut.write("\n")

            #Number RTC
            fOut.write("\n")
            fOut.write("Number Newly RTC")
            fOut.write("\n\t")
            for t in range(self.inputs.runtime):
                if t!= 0:
                    fOut.write("\t"*(len(RISK_STATES)+1))
                fOut.write("Year {}".format(self.inputs.startyear + t))
            fOut.write("\nAge")
            for t in range(self.inputs.runtime):
                if t != 0:
                    fOut.write("\t")
                for j in RISK_STATES:
                    fOut.write("\t"+RISK_STRS[j])
            for i in range(NUM_AGE_GROUPS):
                fOut.write("\n{}".format(i))
                for t in range(self.inputs.runtime):
                    for j in RISK_STATES:
                        fOut.write("\t{}".format(self.outputs.num_rtc[i][j][t]))
                    fOut.write("\t")
            fOut.write("\n")
            
            #Number incident infections
            fOut.write("\n")
            fOut.write("Number Incident Infections")
            fOut.write("\n\t")
            for t in range(self.inputs.runtime):
                if t!= 0:
                    fOut.write("\t"*(len(RISK_STATES)+1))
                fOut.write("Year {}".format(self.inputs.startyear + t))
            fOut.write("\nAge")
            for t in range(self.inputs.runtime):
                if t != 0:
                    fOut.write("\t")
                for j in RISK_STATES:
                    fOut.write("\t"+RISK_STRS[j])
            for i in range(NUM_AGE_GROUPS):
                fOut.write("\n{}".format(i))
                for t in range(self.inputs.runtime):
                    for j in RISK_STATES:
                        fOut.write("\t{}".format(self.outputs.num_inf[i][j][t]))
                    fOut.write("\t")
            fOut.write("\n")      
    def read_inputs(self, filepath):
        """reads inputs from excel file"""
        wb = open_workbook(filepath)

        #read cohort inputs
        cohort_sheet = wb.sheet_by_name('Cohort')

        i = 0
        while True:
            try:
                year = int(cohort_sheet.cell(3,2+i).value)
                self.inputs.num_births.append(float(cohort_sheet.cell(4, 2+i).value))
                if i==0:
                    self.inputs.startyear = year
            except (ValueError,IndexError):
                break
            i+=1
            
        self.inputs.runtime = i
        self.outputs.initialize_outputs(self.inputs.runtime)
        
        #read generated cohort
        generated_sheet = wb.sheet_by_name('Generated Cohort')

        for i in range(NUM_AGE_GROUPS):
            for j in range(len(RISK_STATES)):
                self.inputs.num_prevalent[i][j][IN_CARE] = float(generated_sheet.cell(4+i, 2+j).value)
                self.inputs.num_prevalent[i][j][NEVER_CARE] = float(generated_sheet.cell(4+i, 11+j).value)
                self.inputs.num_prevalent[i][j][LTFU_CARE] = float(generated_sheet.cell(4+i, 20+j).value)
                
        for i in range(NUM_AGE_GROUPS):
            for j in range(self.inputs.runtime):
                self.inputs.num_risk[i][j][PERI_RISK] = 0.0
                self.inputs.num_risk[i][j][MSM_HIGH_RISK] = float(generated_sheet.cell(35+i, 2+j).value)
                self.inputs.num_risk[i][j][MSM_LOW_RISK] = float(generated_sheet.cell(35+i, 15+j).value)
                self.inputs.num_risk[i][j][FSW_RISK] = float(generated_sheet.cell(66+i, 2+j).value)
                self.inputs.num_risk[i][j][IDU_RISK] = float(generated_sheet.cell(66+i, 15+j).value)
                self.inputs.num_risk[i][j][LOW_RISK] = float(generated_sheet.cell(97+i, 2+j).value)

        #read Incidence
        inc_sheet = wb.sheet_by_name('Incidence')
        for i in range(NUM_AGE_GROUPS):
            for j in range(self.inputs.runtime):
                self.inputs.inc_rate[i][j][PERI_RISK] = 0.0
                self.inputs.inc_rate[i][j][MSM_HIGH_RISK] = float(inc_sheet.cell(5+i, 2+j).value)
                self.inputs.inc_rate[i][j][MSM_LOW_RISK] = float(inc_sheet.cell(5+i, 15+j).value)
                self.inputs.inc_rate[i][j][FSW_RISK] = float(inc_sheet.cell(36+i, 2+j).value)
                self.inputs.inc_rate[i][j][IDU_RISK] = float(inc_sheet.cell(36+i, 15+j).value)
                self.inputs.inc_rate[i][j][LOW_RISK] = float(inc_sheet.cell(67+i, 2+j).value)
        
        #read mtct inputs
        mtct_sheet = wb.sheet_by_name('MTCT')
                
        self.inputs.mtct_without_pmtct = []
        for i in range(self.inputs.runtime):
            self.inputs.pmtct.append(
                float(mtct_sheet.cell(4,2+i).value))
            self.inputs.prev_mothers.append(
                float(mtct_sheet.cell(4, 15+i).value))
            self.inputs.bf.append(
                float(mtct_sheet.cell(10,2+i).value))
            self.inputs.mtct_with_pmtct_iuip.append(
                float(mtct_sheet.cell(16,2+i).value))
            self.inputs.mtct_with_pmtct_pp.append(
                float(mtct_sheet.cell(17,2+i).value))
            self.inputs.mtct_without_pmtct_iuip.append(
                float(mtct_sheet.cell(16,15+i).value))
            self.inputs.mtct_without_pmtct_pp.append(
                float(mtct_sheet.cell(17,15+i).value))

        
        #read follow up inputs
        follow_sheet = wb.sheet_by_name('Follow-up')

        risk_cols = (2, 17, 32, 47, 62, 77)
        for i in range(NUM_AGE_GROUPS):
            for j in range(self.inputs.runtime):
                for k in RISK_STATES:
                    self.inputs.diagnosis[i][j][k] = float(follow_sheet.cell(5+i, risk_cols[k]+j).value)
                    self.inputs.linkage[i][j][k] = float(follow_sheet.cell(36+i, risk_cols[k]+j).value)
                    self.inputs.ltfu[i][j][k] = float(follow_sheet.cell(67+i, risk_cols[k]+j).value)
                    self.inputs.rtc[i][j][k] = float(follow_sheet.cell(98+i, risk_cols[k]+j).value)
        
        #read survival inputs
        survival_sheet = wb.sheet_by_name('Survival')
        for i in range(NUM_AGE_GROUPS):
            for j in RISK_STATES:
                for t in range(self.inputs.runtime):
                    if j == PERI_RISK:
                        row = 6
                    else:
                        row = 39
                    self.inputs.survival[i][j][IN_CARE][t] = float(survival_sheet.cell(row+i,2+t).value)
                    self.inputs.survival[i][j][NEVER_CARE][t] = float(survival_sheet.cell(row+i,15+t).value)
                    self.inputs.survival[i][j][LTFU_CARE][t] = float(survival_sheet.cell(row+i,15+t).value)
    def plot1(self, filepath):
        """
        Plots graphs of infected by category

        Hatch Options
        /   - diagonal hatching
        \\   - back diagonal
        |   - vertical
        -   - horizontal
        +   - crossed
        x   - crossed diagonal
        o   - small circle
        O   - large circle
        .   - dots
        *   - stars

        Color Options
        b: blue
        g: green
        r: red
        c: cyan
        m: magenta
        y: yellow
        k: black
        w: white
        """
        #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 0-25 years living with HIV between\n2005-2015 according to care'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Deceased','Not in care', 'In care', 'LTFU', 'Aged out']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        #culmative deaths by year
        num_deaths = np.cumsum(np.sum(self.outputs.num_deaths, axis = (0,1,2)))
        #num alive excluding deaths and ageing out for that year
        num_alive = np.sum(self.outputs.num_alive[:-1] - self.outputs.num_deaths[:-1], axis = (0,1))
        #culmative number aging out by year
        num_age_out = np.cumsum(np.sum(self.outputs.num_alive[-1], axis = (0,1)))

        plot_data = np.array([num_deaths, num_alive[NEVER_CARE], num_alive[IN_CARE],
                     num_alive[LTFU_CARE], num_age_out])

        ind = np.arange(self.inputs.runtime)
        
        plt_bars = [plt.bar(ind, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(plot_data[:i,:], axis = 0))
                    for i, data in enumerate(plot_data)]

        plt.xticks(ind + width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.tick_params(axis='x', which='both', bottom='off')
        plt.title(title, **title_font)
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()
    def plot2(self, filepath):
        """
        Plots graph of Infected by Category with risk group stacking
        """

        #Figure Options
        width = .15
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 0-25 years living with HIV between\n2005-2015 according to care'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Deceased','Not in care', 'In care', 'LTFU', 'Aged out']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        #culmative deaths by risk and year
        num_deaths = np.cumsum(np.sum(self.outputs.num_deaths, axis = (0,2)), axis = 1)
        #num alive excluding deaths and ageing out for that year
        num_alive = np.sum(self.outputs.num_alive[:-1] - self.outputs.num_deaths[:-1], axis = (0,))
        #culmative number aging out by year
        num_age_out = np.cumsum(np.sum(self.outputs.num_alive[-1], axis = (1,)), axis = 1)

        plot_data = [np.array([num_deaths[i], num_alive[i][NEVER_CARE], num_alive[i][IN_CARE],
                     num_alive[i][LTFU_CARE], num_age_out[i]]) for i in RISK_STATES]

        ind = np.arange(self.inputs.runtime)


        
        plt_bars = [[plt.bar(ind+risk*width, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(plot_data[risk][:i,:], axis = 0))
                    for i, data in enumerate(plot_data[risk])] for risk in RISK_STATES]

        plt.xticks(ind + len(RISK_STATES)*width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.title(title, **title_font)

        plt.tick_params(axis='x', which='both', bottom='off')
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        
        plt.savefig(filepath)
        plt.close()
        
    def plot3(self, filepath):
        """
        Plots graphs of infected by Age
        """
        #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 0-25 years living with HIV between\n2005-2015 by age group'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['0-2','2-5', '5-10', '10-15', '15-25']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        #Age buckets do not include right endpoint
        age_buckets = [0,2,5,10,15,25]

        #Num alive by age and year
        num_age = np.sum(self.outputs.num_alive, axis = (1,2))
        #num alive in each age bucket age bucket
        num_age_bucket = np.array([np.sum(num_age[age_buckets[i]:age_buckets[i+1]], axis = (0,)) for i in range(len(age_buckets[:-1]))])

        ind = np.arange(self.inputs.runtime)

        plt_bars = [plt.bar(ind, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(num_age_bucket[:i,:], axis = 0))
                    for i, data in enumerate(num_age_bucket)]

        plt.xticks(ind + width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.tick_params(axis='x', which='both', bottom='off')
        plt.title(title, **title_font)
        
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()
        
    def plot4(self, filepath):
        """
        Plots graphs of infected by proportion at each Age
        """

        #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Proportion of youth aged 0-25 years living with HIV between\n2005-2015 by age group'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['0-2','2-5', '5-10', '10-15', '15-25']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        #Age buckets do not include right endpoint
        age_buckets = [0,2,5,10,15,25]

        #Num alive by age and year
        num_age = np.sum(self.outputs.num_alive, axis = (1,2))
        #num alive in each age bucket age bucket
        num_age_bucket = np.array([np.sum(num_age[age_buckets[i]:age_buckets[i+1]], axis = (0,)) for i in range(len(age_buckets[:-1]))])
        num_prop_age_bucket = np.divide(num_age_bucket, np.sum(num_age_bucket, axis = 0, keepdims = True))

        ind = np.arange(self.inputs.runtime)

        plt_bars = [plt.bar(ind, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(num_prop_age_bucket[:i,:], axis = 0))
                    for i, data in enumerate(num_prop_age_bucket)]

        plt.xticks(ind + width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.tick_params(axis='x', which='both', bottom='off')
        plt.title(title, **title_font)
        
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()

    def plot5(self, filepath):
        """
        Plot Number of people with HIV by risk

        Markers
        '-' 	solid line style
        '--' 	dashed line style
        '-.' 	dash-dot line style
        ':' 	dotted line style
        '.' 	point marker
        ',' 	pixel marker
        'o' 	circle marker
        'v' 	triangle_down marker
        '^' 	triangle_up marker
        '<' 	triangle_left marker
        '>' 	triangle_right marker
        '1' 	tri_down marker
        '2' 	tri_up marker
        '3' 	tri_left marker
        '4' 	tri_right marker
        's' 	square marker
        'p' 	pentagon marker
        '*' 	star marker
        'h' 	hexagon1 marker
        'H' 	hexagon2 marker
        '+' 	plus marker
        'x' 	x marker
        'D' 	diamond marker
        'd' 	thin_diamond marker
        '|' 	vline marker
        '_' 	hline marker
        """

        #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        line_styles = ['s-','*-','+-','1-','h-','D-']
        
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 0-25 years living with HIV between\n2005-2015 by risk group'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Perinatal','MSM High Risk', 'MSM Low Risk' 'FSW', 'IDU', 'Low Risk']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}

        #total num infected by risk and year
        num_inf = np.sum(self.outputs.num_alive, axis = (0,2))

        ind = np.arange(self.inputs.runtime)

        plots = [plt.plot(ind, num_inf[risk], line_styles[risk]) for risk in RISK_STATES]
        
        plt.xticks(ind, ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.title(title, **title_font)
        
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()
        
    def plot6(self, filepath):
        """
        Plot Number of new infections by risk 
        """
        #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]

        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of incident infections in 2005-2015 by risk group'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Perinatal','MSM High Risk', 'MSM Low Risk', 'FSW', 'IDU', 'Low Risk']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        #total num incident infections by risk and year
        num_inf = np.sum(self.outputs.num_inf, axis = (0,))

        ind = np.arange(self.inputs.runtime)
        line_styles = ['s-','*-','+','1-','h-','D-']
        plots = [plt.plot(ind, num_inf[risk], line_styles[risk]) for risk in RISK_STATES]

        plt.xticks(ind, ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.title(title, **title_font)
        
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()

    def plot7(self, filepath):
        """
        Plot bar graph of num infected among certain age group
        """

                #Figure Options
        width = .55
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 15-24 years living with HIV between\n2005-2015 according to care'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Deceased','Not in care', 'In care', 'LTFU', 'Aged out']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        age_range = (15,25)
        #culmative deaths by year
        num_deaths = np.cumsum(np.sum(self.outputs.num_deaths[age_range[0]:age_range[1]], axis = (0,1,2)))
        #num alive excluding deaths and ageing out for that year
        num_alive = np.sum(self.outputs.num_alive[age_range[0]:age_range[1]] - self.outputs.num_deaths[age_range[0]:age_range[1]], axis = (0,1))

        plot_data = np.array([num_deaths, num_alive[NEVER_CARE], num_alive[IN_CARE],
                     num_alive[LTFU_CARE]])

        ind = np.arange(self.inputs.runtime)

        
        plt_bars = [plt.bar(ind, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(plot_data[:i,:], axis = 0))
                    for i, data in enumerate(plot_data)]

        plt.xticks(ind + width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.tick_params(axis='x', which='both', bottom='off')
        plt.title(title, **title_font)
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        plt.savefig(filepath)
        plt.close()
        
    def plot8(self, filepath):
        """
        Plots bar graph of hiv infected among certain age group stacked by risk group
        """
        
        #Figure Options
        width = .15
        colors = ["k","#C0504D","#9BBB59","#8064A2","#FCD5B5"]
        hatches = ["","'","","\\",""]
        x_tick_label_font = {'family':'Calibri', 'style':'normal','size':8.5, 'y':-.01}
        y_tick_label_font = {'family':'Calibri', 'style':'normal','size':10}
        title = 'Number of youth aged 15-24 years living with HIV between\n2005-2015 according to care'
        title_font = {'family':'Calibri', 'weight':'bold', 'size':15, 'y':1.03}
        legend_labels = ['Deceased','Not in care', 'In care', 'LTFU', 'Aged out']
        legend_params = { 'ncol':5, 'loc':'center', 'bbox_to_anchor':(0.5, -.08),
                         'prop':{'size':6}}
        
        age_range = (15,25)
        #culmative deaths by risk and year
        num_deaths = np.cumsum(np.sum(self.outputs.num_deaths[age_range[0]:age_range[1]], axis = (0,2)), axis = 1)
        #num alive excluding deaths and ageing out for that year
        num_alive = np.sum(self.outputs.num_alive[age_range[0]:age_range[1]] - self.outputs.num_deaths[age_range[0]:age_range[1]], axis = (0,))

        plot_data = [np.array([num_deaths[i], num_alive[i][NEVER_CARE], num_alive[i][IN_CARE],
                     num_alive[i][LTFU_CARE]]) for i in RISK_STATES]

        ind = np.arange(self.inputs.runtime)
        
        plt_bars = [[plt.bar(ind+risk*width, data, width, color = colors[i],
                            hatch = hatches[i], bottom = np.sum(plot_data[risk][:i,:], axis = 0))
                    for i, data in enumerate(plot_data[risk])] for risk in RISK_STATES]

        plt.xticks(ind + len(RISK_STATES)*width/2., ["Year "+str(self.inputs.startyear+i) for i in range(self.inputs.runtime)], **x_tick_label_font)
        plt.yticks(**y_tick_label_font)
        plt.tick_params(axis='x', which='both', bottom='off')
        plt.title(title, **title_font)
        
        ax = plt.gca()
        ax.set_xlim(-.5,self.inputs.runtime)

        plt.legend(legend_labels, **legend_params)
        
        plt.savefig(filepath)
        plt.close()
#########################################################
if __name__ == "__main__":
    for input_file in glob("*.xlsx"):
        if input_file.startswith("~"):
            #exclude temp files
            continue
        print(input_file)
        sim = Sim()
        sim.read_inputs(input_file)
        sim.init_run()
        sim.run()
        base_path = os.path.splitext(input_file)[0]
        sim.write_outputs(base_path+".out")
        plot_funcs = ['plot1', 'plot2', 'plot3','plot4', 'plot5', 'plot6', 'plot7', 'plot8']
        for func in plot_funcs:
            getattr(sim, func)(base_path +"_"+func+".png")
