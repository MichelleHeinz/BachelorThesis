################ SIMULATION #################

#############################################
# MAIN FUNCTION
#############################################

# IMPORT
from typing import List, Any
import random
import numpy as np
#import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy

#############################################
# GLOBAL VARS
#############################################

myPref = list()
allPref = list()
alt = list()
debugMode = False

#############################################
# SETTING PLOTS
#############################################

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

#############################################
## METHOD: generate ordered list of alternatives

def altList(m):
    alt = [chr(value) for value in range(ord('a'), ord('a') + (m))]
    return alt
#############################################

#############################################
## METHOD: generate prefProfile

def prefProfile(n, m, alt):

    # preparation
    prefProf = list() # preference profile

    # Generate ordered list of the m alternatives
    #alt = [chr(value) for value in range(ord('a'), ord('a') + (m))]
    #print(alt)

    prefProf.append(['a', 'b', 'c'])
    for i in range(0, n-1):
        # Take a random shuffle from the alt list and append prefRanking to prefProfile
        prefProf.append(random.sample(alt, len(alt)))

    if debugMode:
        print("Preference Profile: {}".format(prefProf))

    return prefProf
#############################################

#############################################
## METHOD: utility score
# input: list of preferences, string singleAlt

def u(myPref, singleAlt):
    m = len(myPref)
    index = myPref.index(singleAlt)
    #print("With myPref {} alternative {} gets {} utility".format(myPref, singleAlt, (m-index-1)))
    return (m - index - 1)

#############################################

#############################################
## METHOD: expected Utility
# input: list of p, list of myPref, list of alt, uType type of utility fct

def expectedUtility(p, myPref, alt, lied, dist, LC1, LC2, uType):
    sum = 0

    # Model 0: u(a)
    if (uType == 0):
        for i in range(0, len(p)):
            sum = sum + (p[i] * u(myPref, alt[i]))

    # Model 1: u(a) - lied*LC
    if (uType == 1):
        for i in range(0, len(p)):
            sum = sum + (p[i] * (u(myPref, alt[i]) - lied * LC1))

    # Model 2: u(a) - dist*LC2
    elif (uType == 2):
        for i in range(0, len(p)):
            sum = sum + (p[i] * (u(myPref, alt[i]) - dist * LC2))

    # Model 3: u(a) - dist*LC2
    elif (uType == 3):
        for i in range(0, len(p)):
            sum = sum + (p[i] * (u(myPref, alt[i]) - lied * LC1 - dist * LC2))

    #print("Expected Utility: {}".format(sum))
    return sum

#############################################

#############################################
## METHOD: majority matrix

def majorityMatrix(allPref, alt):
    M = list()

    # compute for all alternatives a list -> idea: compute scores of one column, then add to M
    for m1 in range(0, len(alt)):
        alt1 = alt[m1]
        colM = list()

        # go through alternatives with alt1 fixed
        for m2 in range(0, len(alt)):
            alt2 = alt[m2]

            # if alt1 = alt2 -> 0
            if alt1 is alt2:
                colM.append(0)
            else:
                # -> NOW: compute score for alt1 vs. alt2
                # go over whole preference profile in order to check how often alt2 > alt1
                scoreCount = 0
                for i in range(0, len(allPref)):
                    # for each list within the big allPref list
                    # if alt1 > alt2 -> +1
                    if allPref[i].index(alt2) > allPref[i].index(alt1):
                        scoreCount = scoreCount-1
                    else:
                        scoreCount = scoreCount+1
                colM.append(scoreCount)

        M.append(colM)

    return M
#############################################

#############################################
## METHOD: compute ML for 3 alternatives

def maximalLottery3(M):
    A = - np.array(M)
    A = A.transpose()
    #print("Negated Matrix {}".format(A))
    b = [0,0,0]
    #print(b)
    A_e = [[1,1,1], [0,0,0], [0,0,0]]
    #print(A_e)
    b_e = [1,0,0]
    res = linprog(c = np.ones(len(A)), A_ub=A, b_ub=b, A_eq=A_e, b_eq=b_e, method='highs')
    #print(res.x)
    return res.x
#############################################

#############################################
## METHOD: compute ML for any number of alternatives

def maximalLottery(M):
    A = - np.array(M)
    A = A.transpose()
    #print("Negated Matrix {}".format(A))
    b = np.zeros((1, len(M)))
    #print(b)
    A_e = np.zeros((len(M), len(M)))
    A_e[0] = A_e[0] + 1
    #print(A_e)
    b_e = np.zeros((1, len(M)))
    b_e[0][0] = 1
    #print(b_e)
    res = linprog(c = np.ones(len(A)), A_ub=A, b_ub=b, A_eq=A_e, b_eq=b_e, method='highs')
    #print(res.x)
    return res.x

#############################################
# compute M majority matrix from two preference profiles
def computeM(M, myPref):

    manMList = []
    distList = []
    M = np.array(M)
    alt = altList(len(myPref))
    # loop over all permutations of myPref
    for man in itertools.permutations(myPref):
        # compare each two elements
        manM = M.copy()
        for i, x in enumerate(alt):
            for j, y in enumerate(alt):
                # compute distance in myPref of pair
                # get index of pair[0] in myPref
                sgn_dist_true = np.sign(myPref.index(x) - myPref.index(y))
                # compute distance in man of pair
                sgn_dist_man = np.sign(man.index(x) - man.index(y))

                # if(sgn_dist_true == sgn_dist_man): M
                if(sgn_dist_true < sgn_dist_man):
                    manM[i, j] = manM[i, j] - 2
                if (sgn_dist_true > sgn_dist_man):
                    manM[i, j] = manM[i, j] + 2
        # add manM to list of Manipulated Majority Matrices
        manMList.append(manM)
        dist = kendall_tau_distance(myPref, man, alt)
        print('')
        print(M)
        print('')
        print("{} Matrix with TruePref: {} and ManPref: {} and dist = {}".format(manM, myPref, man, dist))
    return manMList, distList

def kendall_tau_distance(order_a, order_b, alt):
    pairs = itertools.combinations(alt, 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance

#############################################
## METHOD: Find most beneficial manipulation for 3 alternatives

def findBestManipulation(myPref, M, LC1, LC2, uType):
    #debugMode = False
    M = np.array(M)
    alt = altList(len(myPref))
    myPrefTemp = myPref.copy()

    ## Compute majority matrix & prefProfile of all 5 possible manipulations
    #1 Man: abc - acb
    M1 = M.copy()
    m1 = swap(myPrefTemp, 1, 2)
    M1[1, 2] = M1[1, 2] - 2
    M1[2, 1] = M1[2, 1] + 2
    dist1 = 1
    myPrefTemp = myPref.copy()

    #2 Man: abc - bac
    M2 = M.copy()
    m2 = swap(myPrefTemp, 0, 1)
    M2[0, 1] = M2[0, 1] - 2
    M2[1, 0] = M2[1, 0] + 2
    dist2 = 1
    myPrefTemp = myPref.copy()

    #3 Man: abc - bca
    M3 = M2.copy()
    m3 = swap(m2, 1, 2)
    M3[0, 2] = M3[0, 2] - 2
    M3[2, 0] = M3[2, 0] + 2
    dist3 = 2
    myPrefTemp = myPref.copy()

    #4 Man: abc - cab
    M4 = M1.copy()
    m4 = swap(m1, 0, 1)
    M4[0, 2] = M4[0, 2] - 2
    M4[2, 0] = M4[2, 0] + 2
    dist4 = 2
    myPrefTemp = myPref.copy()

    #5 Man: abc - cba
    M5 = M4.copy()
    m5 = swap(m4, 1, 2)
    M5[0, 1] = M5[0, 1] - 2
    M5[1, 0] = M5[1, 0] + 2
    dist5 = 3
    myPrefTemp = myPref.copy()

    mList = [M, M1, M2, M3, M4, M5]
    distList = [0, dist1, dist2, dist3, dist4, dist5]
    lieList = [0,1,1,1,1,1]
    ## Compute ML p and check whether the manipulation is beneficial

    # compute p
    pList = []
    for Mi in mList:
        p = maximalLottery(Mi)
        pList.append(p)

    # compute expectedUtility for all 5 possible manipulations
    eUList = []
    for i in range(0, len(mList)):
        eU = expectedUtility(pList[i], myPref, alt, lieList[i], distList[i], LC1, LC2, uType)
        eUList.append(eU)

    if debugMode:
        print("** uTrue is {} with p = {} and myPref {}".format(eUList[0], pList[0], myPref))


    betterList = [0,0,0,0,0,0]
    rbetterList = [0,0,0,0,0,0]

    # check: beneficial?
    for i in range(1, len(eUList)):
        if eUList[i] > eUList[0]:
            betterList[i] = eUList[i] - eUList[0]
            if eUList[0] > 0:
                rbetterList[i] = eUList[i] / eUList[0]
            else:
                rbetterList[i] = 0 #TODO: What then?!
        else:
            betterList[i] = 0
            rbetterList[i] = 0

    # print indices of entries in betterList and rbetterlist which are not 0
    successes = []
    for i in range(1, len(betterList)):
        if not betterList[i] == 0:
            successes.append(i)
    
    successes2 = [i for i, e in enumerate(betterList) if e != 0]
    if not successes == successes2:
        print("ATTENTION ONE OF THE CALCULATIONS IS WRONG")

    if debugMode:
        print("Successes: {}".format(successes))


    # check: what is highest absolute gain (difference) in utility?
    diffUtility = max(betterList)
    if diffUtility is not 0:
        # check: which manipulation leads to highest gain?
        maxMan = betterList.index(diffUtility)
        # save abs. & rel utility gain
        absUtility = eUList[maxMan]
        relUtility = rbetterList[maxMan]
    else:
        maxMan = 0
        absUtility = 0
        relUtility = 0


    if debugMode:
        print('***** My PREFERENCES *****')
        print('{}'.format(myPref))
        print('***** True profile: *****')
        print('{} with ML {}'.format(mList[0], pList[0]))
        for i in range(1, len(mList)):
            print('***** Man{}: *****'.format(i))
            print('{} with ML {}'.format(mList[i], pList[i]))

    if debugMode:
        print('**************************************************')
        print("Highest manipulation is No. {}".format(maxMan))
        print('***** utility true:       {}'.format(eUList[0]))
        print('***** absolute utility:   {}'.format(absUtility))
        print('***** utility difference: {}'.format(diffUtility))
        print('***** relative utility:   {}'.format(relUtility))
        print('**************************************************')


    return diffUtility

#############################################
## METHOD: OLD - Find most beneficial manipulation

# def findBestManipulationOLD(myPref, M):

#     M = np.array(M)
#     alt = altList(len(myPref))
#     myPrefTemp = myPref.copy()

#     # Compute majority matrix & prefProfile of all 5 possible manipulations
#     #1 Man: abc - acb
#     M1 = M.copy()
#     m1 = swap(myPrefTemp, 1, 2)
#     M1[1, 2] = M1[1, 2] - 2
#     M1[2, 1] = M1[2, 1] + 2

#     #2 Man: abc - bac
#     M2 = M.copy()
#     m2 = swap(myPrefTemp, 0, 1)
#     M2[0, 1] = M2[0, 1] - 2
#     M2[1, 0] = M2[1, 0] + 2

#     #3 Man: abc - bca
#     M3 = M2.copy()
#     m3 = swap(m2, 1, 2)
#     M3[0, 2] = M3[0, 2] - 2
#     M3[2, 0] = M3[2, 0] + 2

#     #4 Man: abc - cab
#     M4 = M1.copy()
#     m4 = swap(m1, 0, 1)
#     M4[0, 2] = M4[0, 2] - 2
#     M4[2, 0] = M4[2, 0] + 2

#     #5 Man: abc - cba
#     M5 = M4.copy()
#     m5 = swap(m4, 1, 2)
#     M5[0, 1] = M5[0, 1] - 2
#     M5[1, 0] = M5[1, 0] + 2

#     ## Compute ML p and check whether the manipulation is beneficial

#     # compute p
#     p = maximalLottery(M)
#     p1 = maximalLottery(M1)
#     p2 = maximalLottery(M2)
#     p3 = maximalLottery(M3)
#     p4 = maximalLottery(M4)
#     p5 = maximalLottery(M5)

#     print('***** True profile: *****')
#     print('{} with ML {}'.format(M, p))
#     print('***** Man1: *****')
#     print('{} with ML {}'.format(M1, p1))
#     print('***** Man2: *****')
#     print('{} with ML {}'.format(M2, p2))
#     print('***** Man3: *****')
#     print('{} with ML {}'.format(M3, p3))
#     print('***** Man4: *****')
#     print('{} with ML {}'.format(M4, p4))
#     print('***** MAn5: *****')
#     print('{} with ML {}'.format(M5, p5))

#     # check: beneficial?
#     uTrue = expectedUtility(p, myPref, alt)
#     print("** uTrue is {} with p = {} and myPref {}".format(uTrue, p, myPref))
#     expU1 = expectedUtility(p1, myPref, alt)
#     expU2 = expectedUtility(p2, myPref, alt)
#     expU3 = expectedUtility(p3, myPref, alt)
#     expU4 = expectedUtility(p4, myPref, alt)
#     expU5 = expectedUtility(p5, myPref, alt)

#     better1 = better2 = better3 = better4 = better5 = 0
#     rbetter1 = rbetter2 = rbetter3 = rbetter4 = rbetter5 = 0

#     if expU1 > uTrue:
#         better1 = expU1 - uTrue
#         rbetter1 = expU1 / uTrue
#         print("***** Success: M1 beneficial *****")
#         print(better1)
#     if expU2 > uTrue:
#         better2 = expU2 - uTrue
#         rbetter2 = expU2 / uTrue
#         print("***** Success: M2 beneficial *****")
#         print(better2)
#     if expU3 > uTrue:
#         better3 = expU3 - uTrue
#         rbetter3 = expU3 / uTrue
#         print("***** Success: M3 beneficial *****")
#         print(better3)
#     if expU4 > uTrue:
#         better4 = expU4 - uTrue
#         rbetter4 = expU4 / uTrue
#         print("***** Success: M4 beneficial *****")
#         print(better4)
#     if expU5 > uTrue:
#         better5 = expU5 - uTrue
#         rbetter5 = expU5 / uTrue
#         print("***** Success: M5 beneficial *****")
#         print(better5)

#     betterList = [better1, better2, better3, better4, better5]
#     rbetterList = [rbetter1, rbetter2, rbetter3, rbetter4, rbetter5]
#     expUList = [expU1, expU2, expU3, expU4, expU5]
#     diffUtility = max(betterList)
#     if diffUtility is not 0:
#         maxMan = betterList.index(diffUtility) + 1
#         absUtility = expUList[maxMan - 1]
#         relUtility = rbetterList[maxMan - 1]
#     else:
#         maxMan = 0
#         absUtility = 0
#         relUtility = 0

#     print('**************************************************')
#     print("Highest manipulation is No. {}".format(maxMan))
#     print('***** utility true:       {}'.format(uTrue))
#     print('***** absolute utility:   {}'.format(absUtility))
#     print('***** utility difference: {}'.format(diffUtility))
#     print('***** relative utility:   {}'.format(relUtility))
#     print('**************************************************')

#     return diffUtility

## Swap function
def swap(list, pos1, pos2):

    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

#############################################

#############################################
## METHOD: Simulation for Model 1
# input: size = size of simulation, n #voters, m #alt

def simulationModel1(n, m, size, lcDist):

    alt = altList(m)
    # for each prefProfile save how beneficial the best manipulation compared to uTrue is

    # create list of lists
    masterList = []
    # idea: sample for every fix lc (i + lcDist) size-many profiles
    i = 0
    while i < 1.5:
        masterList.append([])
        i += lcDist

    print('***** Simulation for Model 1 *****')
    print('Length of masterList: {}'.format(len(masterList)))


    # create i times a new prefProfile
    for i in range(size):
        allPref = prefProfile(n, m, alt)
        myPref = allPref[0]
        M = majorityMatrix(allPref, alt)

        for i in range(len(masterList)):
            masterList[i].append(findBestManipulation(myPref, M, i*lcDist, 0, 1))

    # loop over all lists and count nonzero elements
    numberManList = []
    for i in range(len(masterList)):
        numberManList.append(np.count_nonzero(masterList[i]))

    #if debugMode:
    print('')
    print("number of manipulations for each LC:")
    print(numberManList)
    print('')

    return numberManList


def simulationModel1vs2(n, m, size, lcDist):

    alt = altList(m)
    # for each prefProfile save how beneficial the best manipulation compared to uTrue is

    # create list of lists
    m1_masterList = []
    m2_masterList = []
    # idea: sample for every fix lc (i + lcDist) size-many profiles
    i = 0
    while i < 0.6:
        m1_masterList.append([])
        m2_masterList.append([])
        i += lcDist

    print('***** Simulation for Model 1 & 2 *****')
    print('Length of masterList: {}'.format(len(m1_masterList)))

    # create i times a new prefProfile
    for i in range(size):
        allPref = prefProfile(n, m, alt)
        myPref = allPref[0]
        M = majorityMatrix(allPref, alt)

        for i in range(len(m1_masterList)):
            m1_masterList[i].append(findBestManipulation(myPref, M, i * lcDist, 0, 1))
            m2_masterList[i].append(findBestManipulation(myPref, M, 0, i * lcDist, 2))

    # loop over all lists and count nonzero elements
    m1_numberManList = []
    m2_numberManList = []
    for i in range(len(m1_masterList)):
        m1_numberManList.append(np.count_nonzero(m1_masterList[i]))
        m2_numberManList.append(np.count_nonzero(m2_masterList[i]))

    # if debugMode:
    print('')
    print("MODEL 1: number of manipulations for each LC:")
    print(m1_numberManList)
    print('')
    print("MODEL 2: number of manipulations for each LC:")
    print(m2_numberManList)
    print('')

    return m1_numberManList, m2_numberManList

#############################################
# MAIN

def test1_general():
    allPref = prefProfile(5,3)
    myPref = allPref[0]
    alt = altList(3)

    print('**********************')
    print('MyPref: ')
    print(myPref)
    print('AllPref: ')
    print(allPref)
    print('Alternatives: ')
    print(alt)

    M = majorityMatrix(allPref, alt)

    print('Majority Margin Matrix: ')
    print(M)


def test2_casesHighestDiff():
    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 1: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -1], [-3, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = - np.array([[0, 3, -1], [-3, 0, 1], [1, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array([[0, 1, -3], [-1, 0, 1], [3, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = [[0, 3, -1], [-3, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 16: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M16 = - np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M16)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M16, 0.4, 0, 0)

def test2_bestManipulation():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 1: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -1], [-3, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 1, -1], [-1, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 1, -3], [-1, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = - np.array([[0, 3, -1], [-3, 0, 1], [1, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 5: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array([[0, 1, -1], [-1, 0, 3], [1, -3, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array([[0, 1, -3], [-1, 0, 1], [3, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = [[0, 3, -1], [-3, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = [[0, 1, -3], [-1, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 9: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M9 = [[0, 3, -3], [-3, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M9)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M9, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 10: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M10 = - np.array(M7)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M10)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M10, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 11: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M11 = - np.array(M8)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M11)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M11, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 12: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M12 = - np.array(M9)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M12)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M12, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 13: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M13 = [[0, 3, -3], [-3, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M13)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M13, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 14: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M14 = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M14)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M14, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 15: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M15 = - np.array(M13)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M15)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M15, 0.4, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 16: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M16 = - np.array(M14)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M16)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M16, 0.4, 0, 0)


def test3_bestManipulationFor14():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -1], [-3, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 5, -1], [-5, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 3, -1], [-3, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 3, -3], [-3, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array([[0, 3, -1], [-3, 0, 1], [1, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array([[0, 5, -1], [-5, 0, 1], [1, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array([[0, 3, -1], [-3, 0, 3], [1, -3, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array([[0, 3, -3], [-3, 0, 1], [3, -1, 0]])
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)

def test3_bestManipulationFor25():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 1, -1], [-1, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 3, -1], [-3, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 1, -1], [-1, 0, 5], [1, -5, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 1, -3], [-1, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array(M1)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array(M2)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array(M3)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array(M4)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)


def test3_bestManipulationFor36():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 1, -3], [-1, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 3, -3], [-3, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 1, -3], [-1, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 1, -5], [-1, 0, 3], [5, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array(M1)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array(M2)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array(M3)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array(M4)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)


def test3_bestManipulationFor710():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL 1 ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -1], [-3, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 5, -1], [-5, 0, 3], [1, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 3, -1], [-3, 0, 5], [1, -5, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 3, -3], [-3, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array(M1)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array(M2)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array(M3)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array(M4)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)

def test3_bestManipulationFor811():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL 1 ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 1, -3], [-1, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 3, -3], [-3, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 1, -3], [-1, 0, 5], [3, -5, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 1, -5], [-1, 0, 3], [5, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array(M1)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array(M2)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array(M3)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array(M4)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)

def test3_bestManipulationFor912():

    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 1: ORIGINAL 1 ++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -3], [-3, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M1, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 2: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M2 = [[0, 5, -3], [-5, 0, 1], [3, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M2)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M2, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 3: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M3 = [[0, 3, -5], [-3, 0, 1], [5, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M3)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M3, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 4: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M4 = [[0, 3, -3], [-3, 0, 3], [3, -3, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M4)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M4, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No. 5: ORIGINAL 2 ++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M5 = - np.array(M1)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M5)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M5, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 6: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M6 = - np.array(M2)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M6)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M6, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 7: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M7 = - np.array(M3)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M7)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M7, 0, 0, 0)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 8: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M8 = - np.array(M4)
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M8)
    print('My true preferences: ')
    print(myPref)
    findBestManipulation(myPref, M8, 0, 0, 0)


def test4_simulationModel():
    lcDist = 0.05
    yValues = simulationModel1(9, 3, 1000, lcDist)

    # PLOT
    plt.figure(figsize=(10, 6), tight_layout=True)
    # plotting
    xValues = []
    for i in range(len(yValues)):
        xValues.append(i * lcDist)
    plt.plot(xValues, yValues, 'o-', linewidth=2)
    # customization
    plt.xlabel('Lying Cost')
    plt.ylabel('Number of manipulable profiles')
    plt.title('Simulation: Manipulable Profiles with varying Lying Cost')
    plt.show()

def test4_simulationModel1vs2():
    lcDist = 0.05
    yValues = simulationModel1vs2(9, 3, 3000, lcDist)
    m1_yValues = yValues[0]
    m2_yValues = yValues[1]

    # PLOT
    plt.figure(figsize=(10, 6), tight_layout=True)
    # plotting
    xValues = []
    for i in range(len(m1_yValues)):
        xValues.append(i * lcDist)
    plt.plot(xValues, m1_yValues, 'o-', linewidth=2, label ='Model 1')
    plt.plot(xValues, m2_yValues, 'x-', linewidth=2, label = 'Model 2')
    # customization
    plt.xlabel('Lying Cost')
    plt.ylabel('Number of manipulable profiles')
    plt.title('Simulation: Manipulable Profiles with varying Lying Cost')
    plt.legend()
    plt.show()

def test4_simulationModel1_differentN():
    lcDist = 0.1
    # PLOT
    plt.figure(figsize=(10, 6), tight_layout=True)
    # plot for increasing n
    for n in range(3, 10):
        yValues = simulationModel1(n, 3, 2000, lcDist)
        # plotting
        xValues = []
        for i in range(len(yValues)):
            xValues.append(i * lcDist)
        plt.plot(xValues, yValues, 'o-', linewidth=2, label = 'n = ' + str(n))

    # customization
    plt.xlabel('Lying Cost')
    plt.ylabel('Number of manipulable profiles')
    plt.title('Simulation: Manipulable Profiles with varying Lying Cost')
    plt.legend()
    plt.show()

def test5_computeM():
    # First circle
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++ No 1: Circle +++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    M1 = [[0, 3, -1], [-3, 0, 1], [1, -1, 0]]
    myPref = ['a', 'b', 'c']
    print('Majority Matrix: ')
    print(M1)
    print('My true preferences: ')
    print(myPref)
    computeM(M1, myPref)


## Choose TEST

#test2_bestManipulation()
#test2_casesHighestDiff()
#test3_bestManipulationFor14()
#test3_bestManipulationFor25()
#test3_bestManipulationFor36()
#test4_simulationModel()
#test4_simulationModel1vs2()
#test4_simulationModel1_differentN()
test5_computeM()