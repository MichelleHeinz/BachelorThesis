################ SIMULATION #################

#############################################
# MAIN FUNCTION
#############################################

# IMPORT
from typing import List, Any
import random
import numpy as np
import string
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp
from scipy.optimize import linprog

#############################################
# GLOBAL VARS
#############################################

myPref = list()
allPref = list()
alt = list()

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

    for i in range(0, n):
        # Take a random shuffle from the alt list and append prefRanking to prefProfile
        prefProf.append(random.sample(alt, len(alt)))

    print(prefProf)
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

#############################################
## METHOD: Find most beneficial manipulation for 3 alternatives

def findBestManipulation(myPref, M, LC1, LC2, uType):

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

    #2 Man: abc - bac
    M2 = M.copy()
    m2 = swap(myPrefTemp, 0, 1)
    M2[0, 1] = M2[0, 1] - 2
    M2[1, 0] = M2[1, 0] + 2
    dist2 = 1

    #3 Man: abc - bca
    M3 = M2.copy()
    m3 = swap(m2, 1, 2)
    M3[0, 2] = M3[0, 2] - 2
    M3[2, 0] = M3[2, 0] + 2
    dist3 = 2

    #4 Man: abc - cab
    M4 = M1.copy()
    m4 = swap(m1, 0, 1)
    M4[0, 2] = M4[0, 2] - 2
    M4[2, 0] = M4[2, 0] + 2
    dist4 = 2

    #5 Man: abc - cba
    M5 = M4.copy()
    m5 = swap(m4, 1, 2)
    M5[0, 1] = M5[0, 1] - 2
    M5[1, 0] = M5[1, 0] + 2
    dist5 = 3

    ## Compute ML p and check whether the manipulation is beneficial

    # compute p
    p = maximalLottery(M)
    p1 = maximalLottery(M1)
    p2 = maximalLottery(M2)
    p3 = maximalLottery(M3)
    p4 = maximalLottery(M4)
    p5 = maximalLottery(M5)

    print('***** True profile: *****')
    print('{} with ML {}'.format(M, p))
    print('***** Man1: *****')
    print('{} with ML {}'.format(M1, p1))
    print('***** Man2: *****')
    print('{} with ML {}'.format(M2, p2))
    print('***** Man3: *****')
    print('{} with ML {}'.format(M3, p3))
    print('***** Man4: *****')
    print('{} with ML {}'.format(M4, p4))
    print('***** MAn5: *****')
    print('{} with ML {}'.format(M5, p5))

    # compute expectedUtility
    uTrue = expectedUtility(p, myPref, alt, 0, 0, LC1, LC2, uType)
    print("** uTrue is {} with p = {} and myPref {}".format(uTrue, p, myPref))
    expU1 = expectedUtility(p1, myPref, alt, 1, dist1, LC1, LC2, uType)
    expU2 = expectedUtility(p2, myPref, alt, 1, dist2, LC1, LC2, uType)
    expU3 = expectedUtility(p3, myPref, alt, 1, dist3, LC1, LC2, uType)
    expU4 = expectedUtility(p4, myPref, alt, 1, dist4, LC1, LC2, uType)
    expU5 = expectedUtility(p5, myPref, alt, 1, dist5, LC1, LC2, uType)

    # initialize
    better1 = better2 = better3 = better4 = better5 = 0
    rbetter1 = rbetter2 = rbetter3 = rbetter4 = rbetter5 = 0

    # check: beneficial?
    if expU1 > uTrue:
        better1 = expU1 - uTrue
        rbetter1 = expU1 / uTrue
        print("***** Success: M1 beneficial *****")
        print(better1)
    if expU2 > uTrue:
        better2 = expU2 - uTrue
        rbetter2 = expU2 / uTrue
        print("***** Success: M2 beneficial *****")
        print(better2)
    if expU3 > uTrue:
        better3 = expU3 - uTrue
        rbetter3 = expU3 / uTrue
        print("***** Success: M3 beneficial *****")
        print(better3)
    if expU4 > uTrue:
        better4 = expU4 - uTrue
        rbetter4 = expU4 / uTrue
        print("***** Success: M4 beneficial *****")
        print(better4)
    if expU5 > uTrue:
        better5 = expU5 - uTrue
        rbetter5 = expU5 / uTrue
        print("***** Success: M5 beneficial *****")
        print(better5)

    # fill lists
    betterList = [better1, better2, better3, better4, better5]
    rbetterList = [rbetter1, rbetter2, rbetter3, rbetter4, rbetter5]
    expUList = [expU1, expU2, expU3, expU4, expU5]

    # check: what is highest absolute gain (difference) in utility?
    diffUtility = max(betterList)
    if diffUtility is not 0:
        # check: which manipulation leads to highest gain?
        maxMan = betterList.index(diffUtility) + 1
        # save abs. & rel utility gain
        absUtility = expUList[maxMan - 1]
        relUtility = rbetterList[maxMan - 1]
    else:
        maxMan = 0
        absUtility = 0
        relUtility = 0

    print('**************************************************')
    print("Highest manipulation is No. {}".format(maxMan))
    print('***** utility true:       {}'.format(uTrue))
    print('***** absolute utility:   {}'.format(absUtility))
    print('***** utility difference: {}'.format(diffUtility))
    print('***** relative utility:   {}'.format(relUtility))
    print('**************************************************')

    return diffUtility

#############################################
## METHOD: OLD - Find most beneficial manipulation

def findBestManipulationOLD(myPref, M):

    M = np.array(M)
    alt = altList(len(myPref))
    myPrefTemp = myPref.copy()

    # Compute majority matrix & prefProfile of all 5 possible manipulations
    #1 Man: abc - acb
    M1 = M.copy()
    m1 = swap(myPrefTemp, 1, 2)
    M1[1, 2] = M1[1, 2] - 2
    M1[2, 1] = M1[2, 1] + 2

    #2 Man: abc - bac
    M2 = M.copy()
    m2 = swap(myPrefTemp, 0, 1)
    M2[0, 1] = M2[0, 1] - 2
    M2[1, 0] = M2[1, 0] + 2

    #3 Man: abc - bca
    M3 = M2.copy()
    m3 = swap(m2, 1, 2)
    M3[0, 2] = M3[0, 2] - 2
    M3[2, 0] = M3[2, 0] + 2

    #4 Man: abc - cab
    M4 = M1.copy()
    m4 = swap(m1, 0, 1)
    M4[0, 2] = M4[0, 2] - 2
    M4[2, 0] = M4[2, 0] + 2

    #5 Man: abc - cba
    M5 = M4.copy()
    m5 = swap(m4, 1, 2)
    M5[0, 1] = M5[0, 1] - 2
    M5[1, 0] = M5[1, 0] + 2

    ## Compute ML p and check whether the manipulation is beneficial

    # compute p
    p = maximalLottery(M)
    p1 = maximalLottery(M1)
    p2 = maximalLottery(M2)
    p3 = maximalLottery(M3)
    p4 = maximalLottery(M4)
    p5 = maximalLottery(M5)

    print('***** True profile: *****')
    print('{} with ML {}'.format(M, p))
    print('***** Man1: *****')
    print('{} with ML {}'.format(M1, p1))
    print('***** Man2: *****')
    print('{} with ML {}'.format(M2, p2))
    print('***** Man3: *****')
    print('{} with ML {}'.format(M3, p3))
    print('***** Man4: *****')
    print('{} with ML {}'.format(M4, p4))
    print('***** MAn5: *****')
    print('{} with ML {}'.format(M5, p5))

    # check: beneficial?
    uTrue = expectedUtility(p, myPref, alt)
    print("** uTrue is {} with p = {} and myPref {}".format(uTrue, p, myPref))
    expU1 = expectedUtility(p1, myPref, alt)
    expU2 = expectedUtility(p2, myPref, alt)
    expU3 = expectedUtility(p3, myPref, alt)
    expU4 = expectedUtility(p4, myPref, alt)
    expU5 = expectedUtility(p5, myPref, alt)

    better1 = better2 = better3 = better4 = better5 = 0
    rbetter1 = rbetter2 = rbetter3 = rbetter4 = rbetter5 = 0

    if expU1 > uTrue:
        better1 = expU1 - uTrue
        rbetter1 = expU1 / uTrue
        print("***** Success: M1 beneficial *****")
        print(better1)
    if expU2 > uTrue:
        better2 = expU2 - uTrue
        rbetter2 = expU2 / uTrue
        print("***** Success: M2 beneficial *****")
        print(better2)
    if expU3 > uTrue:
        better3 = expU3 - uTrue
        rbetter3 = expU3 / uTrue
        print("***** Success: M3 beneficial *****")
        print(better3)
    if expU4 > uTrue:
        better4 = expU4 - uTrue
        rbetter4 = expU4 / uTrue
        print("***** Success: M4 beneficial *****")
        print(better4)
    if expU5 > uTrue:
        better5 = expU5 - uTrue
        rbetter5 = expU5 / uTrue
        print("***** Success: M5 beneficial *****")
        print(better5)

    betterList = [better1, better2, better3, better4, better5]
    rbetterList = [rbetter1, rbetter2, rbetter3, rbetter4, rbetter5]
    expUList = [expU1, expU2, expU3, expU4, expU5]
    diffUtility = max(betterList)
    if diffUtility is not 0:
        maxMan = betterList.index(diffUtility) + 1
        absUtility = expUList[maxMan - 1]
        relUtility = rbetterList[maxMan - 1]
    else:
        maxMan = 0
        absUtility = 0
        relUtility = 0

    print('**************************************************')
    print("Highest manipulation is No. {}".format(maxMan))
    print('***** utility true:       {}'.format(uTrue))
    print('***** absolute utility:   {}'.format(absUtility))
    print('***** utility difference: {}'.format(diffUtility))
    print('***** relative utility:   {}'.format(relUtility))
    print('**************************************************')

    return diffUtility

## Swap function
def swap(list, pos1, pos2):

    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

#############################################

#############################################
## METHOD: Simulation for Model 1
# input: size = size of simulation, n #voters, m #alt

def simulationModel1(n, m, size):

    alt = altList(m)
    # for each prefProfile save how beneficial the best manipulation compared to uTrue is
    diffList00 = list()
    diffList01 = list()
    diffList02 = list()
    diffList03 = list()
    diffList04 = list()
    diffList05 = list()
    diffList06 = list()
    diffList07 = list()
    diffList08 = list()
    diffList09 = list()
    diffList10 = list()
    diffLists = []
    diffLists[0] = 1


    # create i times a new prefProfile
    for i in range(size):
        allPref = prefProfile(n, m, alt)
        myPref = allPref[0]
        M = majorityMatrix(allPref, alt)

        for i in range(10):
            diffLists[i].append

        # no LC1
        diff00 = findBestManipulation(myPref, M, 0, 0, 1)
        diffList00.append(diff00)

        # LC1 = 0.1
        diff01 = findBestManipulation(myPref, M, 0.1, 0, 1)
        diffList01.append(diff01)

        # LC1 = 0.2
        diff02 = findBestManipulation(myPref, M, 0.2, 0, 1)
        diffList02.append(diff02)

        # LC1 = 0.3
        diff03 = findBestManipulation(myPref, M, 0.3, 0, 1)
        diffList03.append(diff03)

        # LC1 = 0.4
        diff04 = findBestManipulation(myPref, M, 0.4, 0, 1)
        diffList04.append(diff04)

        # LC1 = 0.5
        diff05 = findBestManipulation(myPref, M, 0.5, 0, 1)
        diffList05.append(diff05)

        # LC1 = 0.6
        diff06 = findBestManipulation(myPref, M, 0.6, 0, 1)
        diffList06.append(diff06)

        # LC1 = 0.7
        diff07 = findBestManipulation(myPref, M, 0.7, 0, 1)
        diffList07.append(diff07)

        # LC1 = 0.8
        diff08 = findBestManipulation(myPref, M, 0.8, 0, 1)
        diffList08.append(diff08)

        # LC1 = 0.9
        diff09 = findBestManipulation(myPref, M, 0.9, 0, 1)
        diffList09.append(diff09)

        # LC1 = 1.0
        diff10 = findBestManipulation(myPref, M, 1, 0, 1)
        diffList10.append(diff10)

    diffArray00 = np.array(diffList00)
    diffArray01 = np.array(diffList01)
    diffArray02 = np.array(diffList02)
    diffArray03 = np.array(diffList03)
    diffArray04 = np.array(diffList04)
    diffArray05 = np.array(diffList05)
    diffArray06 = np.array(diffList06)
    diffArray07 = np.array(diffList07)
    diffArray08 = np.array(diffList08)
    diffArray09 = np.array(diffList09)
    diffArray10 = np.array(diffList10)

    numMan00 = np.count_nonzero(diffArray00)
    numMan01 = np.count_nonzero(diffArray01)
    numMan02 = np.count_nonzero(diffArray02)
    numMan03 = np.count_nonzero(diffArray03)
    numMan04 = np.count_nonzero(diffArray04)
    numMan05 = np.count_nonzero(diffArray05)
    numMan06 = np.count_nonzero(diffArray06)
    numMan07 = np.count_nonzero(diffArray07)
    numMan08 = np.count_nonzero(diffArray08)
    numMan09 = np.count_nonzero(diffArray09)
    numMan10 = np.count_nonzero(diffArray10)

    print('')
    print('')
    i = 0.0
    diffLists[i]["LC"] =
    for diffList in diffLists:
        print('With LC = {}:: {}'.format(i, diffList))
        i += 0.1
    print('With LC = 0.0:: {}'.format(diffList00))
    print('With LC = 0.1:: {}'.format(diffList01))
    print('With LC = 0.2:: {}'.format(diffList02))
    print('With LC = 0.3:: {}'.format(diffList03))
    print('With LC = 0.4:: {}'.format(diffList04))
    print('With LC = 0.5:: {}'.format(diffList05))
    print('With LC = 0.6:: {}'.format(diffList06))
    print('With LC = 0.7:: {}'.format(diffList07))
    print('With LC = 0.8:: {}'.format(diffList08))
    print('With LC = 0.9:: {}'.format(diffList09))
    print('With LC = 1.0:: {}'.format(diffList10))
    print('')
    print('With LC = 0.0: {} profiles with beneficial manipulations.'.format(numMan00))
    print('With LC = 0.1: {} profiles with beneficial manipulations.'.format(numMan01))
    print('With LC = 0.2: {} profiles with beneficial manipulations.'.format(numMan02))
    print('With LC = 0.3: {} profiles with beneficial manipulations.'.format(numMan03))
    print('With LC = 0.4: {} profiles with beneficial manipulations.'.format(numMan04))
    print('With LC = 0.5: {} profiles with beneficial manipulations.'.format(numMan05))
    print('With LC = 0.6: {} profiles with beneficial manipulations.'.format(numMan06))
    print('With LC = 0.7: {} profiles with beneficial manipulations.'.format(numMan07))
    print('With LC = 0.8: {} profiles with beneficial manipulations.'.format(numMan08))
    print('With LC = 0.9: {} profiles with beneficial manipulations.'.format(numMan09))
    print('With LC = 1.0: {} profiles with beneficial manipulations.'.format(numMan10))

def simulationModel1vs2(n, m, size):

    alt = altList(m)
    # for each prefProfile save how beneficial the best manipulation compared to uTrue is
    m1_diffList00 = list()
    m1_diffList01 = list()
    m1_diffList02 = list()
    m1_diffList03 = list()
    m1_diffList04 = list()
    m1_diffList05 = list()
    m1_diffList06 = list()
    m1_diffList07 = list()
    m1_diffList08 = list()
    m1_diffList09 = list()
    m1_diffList10 = list()

    m2_diffList00 = list()
    m2_diffList01 = list()
    m2_diffList02 = list()
    m2_diffList03 = list()
    m2_diffList04 = list()
    m2_diffList05 = list()
    m2_diffList06 = list()
    m2_diffList07 = list()
    m2_diffList08 = list()
    m2_diffList09 = list()
    m2_diffList10 = list()


    # create i times a new prefProfile
    for i in range(size):
        allPref = prefProfile(n, m, alt)
        myPref = allPref[0]
        M = majorityMatrix(allPref, alt)

        ## MODEL 1
        # no LC1
        diff00 = findBestManipulation(myPref, M, 0, 0, 1)
        m1_diffList00.append(diff00)

        # LC1 = 0.1
        diff01 = findBestManipulation(myPref, M, 0.1, 0, 1)
        m1_diffList01.append(diff01)

        # LC1 = 0.2
        diff02 = findBestManipulation(myPref, M, 0.2, 0, 1)
        m1_diffList02.append(diff02)

        # LC1 = 0.3
        diff03 = findBestManipulation(myPref, M, 0.3, 0, 1)
        m1_diffList03.append(diff03)

        # LC1 = 0.4
        diff04 = findBestManipulation(myPref, M, 0.4, 0, 1)
        m1_diffList04.append(diff04)

        # LC1 = 0.5
        diff05 = findBestManipulation(myPref, M, 0.5, 0, 1)
        m1_diffList05.append(diff05)

        # LC1 = 0.6
        diff06 = findBestManipulation(myPref, M, 0.6, 0, 1)
        m1_diffList06.append(diff06)

        # LC1 = 0.7
        diff07 = findBestManipulation(myPref, M, 0.7, 0, 1)
        m1_diffList07.append(diff07)

        # LC1 = 0.8
        diff08 = findBestManipulation(myPref, M, 0.8, 0, 1)
        m1_diffList08.append(diff08)

        # LC1 = 0.9
        diff09 = findBestManipulation(myPref, M, 0.9, 0, 1)
        m1_diffList09.append(diff09)

        # LC1 = 1.0
        diff10 = findBestManipulation(myPref, M, 1, 0, 1)
        m1_diffList10.append(diff10)

        ## MODEL 2
        # no LC1
        diff00 = findBestManipulation(myPref, M, 0, 0, 2)
        m2_diffList00.append(diff00)

        # LC1 = 0.1
        diff01 = findBestManipulation(myPref, M, 0, 0.1, 2)
        m2_diffList01.append(diff01)

        # LC1 = 0.2
        diff02 = findBestManipulation(myPref, M, 0, 0.2, 2)
        m2_diffList02.append(diff02)

        # LC1 = 0.3
        diff02 = findBestManipulation(myPref, M, 0, 0.3, 2)
        m2_diffList03.append(diff03)

        # LC1 = 0.4
        diff04 = findBestManipulation(myPref, M, 0, 0.4, 2)
        m2_diffList04.append(diff04)

        # LC1 = 0.5
        diff05 = findBestManipulation(myPref, M, 0, 0.5, 2)
        m2_diffList05.append(diff05)

        # LC1 = 0.6
        diff06 = findBestManipulation(myPref, M, 0, 0.6, 2)
        m2_diffList06.append(diff06)

        # LC1 = 0.7
        diff07 = findBestManipulation(myPref, M, 0, 0.7, 2)
        m2_diffList07.append(diff07)

        # LC1 = 0.8
        diff08 = findBestManipulation(myPref, M, 0, 0.8, 2)
        m2_diffList08.append(diff08)

        # LC1 = 0.9
        diff09 = findBestManipulation(myPref, M, 0, 0.9, 2)
        m2_diffList09.append(diff09)

        # LC1 = 1.0
        diff10 = findBestManipulation(myPref, M, 0, 1, 2)
        m2_diffList10.append(diff10)

    m1_diffArray00 = np.array(m1_diffList00)
    m1_diffArray01 = np.array(m1_diffList01)
    m1_diffArray02 = np.array(m1_diffList02)
    m1_diffArray03 = np.array(m1_diffList03)
    m1_diffArray04 = np.array(m1_diffList04)
    m1_diffArray05 = np.array(m1_diffList05)
    m1_diffArray06 = np.array(m1_diffList06)
    m1_diffArray07 = np.array(m1_diffList07)
    m1_diffArray08 = np.array(m1_diffList08)
    m1_diffArray09 = np.array(m1_diffList09)
    m1_diffArray10 = np.array(m1_diffList10)

    m1_numMan00 = np.count_nonzero(m1_diffArray00)
    m1_numMan01 = np.count_nonzero(m1_diffArray01)
    m1_numMan02 = np.count_nonzero(m1_diffArray02)
    m1_numMan03 = np.count_nonzero(m1_diffArray03)
    m1_numMan04 = np.count_nonzero(m1_diffArray04)
    m1_numMan05 = np.count_nonzero(m1_diffArray05)
    m1_numMan06 = np.count_nonzero(m1_diffArray06)
    m1_numMan07 = np.count_nonzero(m1_diffArray07)
    m1_numMan08 = np.count_nonzero(m1_diffArray08)
    m1_numMan09 = np.count_nonzero(m1_diffArray09)
    m1_numMan10 = np.count_nonzero(m1_diffArray10)

    m2_diffArray00 = np.array(m2_diffList00)
    m2_diffArray01 = np.array(m2_diffList01)
    m2_diffArray02 = np.array(m2_diffList02)
    m2_diffArray03 = np.array(m2_diffList03)
    m2_diffArray04 = np.array(m2_diffList04)
    m2_diffArray05 = np.array(m2_diffList05)
    m2_diffArray06 = np.array(m2_diffList06)
    m2_diffArray07 = np.array(m2_diffList07)
    m2_diffArray08 = np.array(m2_diffList08)
    m2_diffArray09 = np.array(m2_diffList09)
    m2_diffArray10 = np.array(m2_diffList10)

    m2_numMan00 = np.count_nonzero(m2_diffArray00)
    m2_numMan01 = np.count_nonzero(m2_diffArray01)
    m2_numMan02 = np.count_nonzero(m2_diffArray02)
    m2_numMan03 = np.count_nonzero(m2_diffArray03)
    m2_numMan04 = np.count_nonzero(m2_diffArray04)
    m2_numMan05 = np.count_nonzero(m2_diffArray05)
    m2_numMan06 = np.count_nonzero(m2_diffArray06)
    m2_numMan07 = np.count_nonzero(m2_diffArray07)
    m2_numMan08 = np.count_nonzero(m2_diffArray08)
    m2_numMan09 = np.count_nonzero(m2_diffArray09)
    m2_numMan10 = np.count_nonzero(m2_diffArray10)

    print('')
    print('*********** MODEL 1 ************')
    print('With LC = 0.0: {}'.format(m1_diffList00))
    print('With LC = 0.1: {}'.format(m1_diffList01))
    print('With LC = 0.2: {}'.format(m1_diffList02))
    print('With LC = 0.3: {}'.format(m1_diffList03))
    print('With LC = 0.4: {}'.format(m1_diffList04))
    print('With LC = 0.5: {}'.format(m1_diffList05))
    print('With LC = 0.6: {}'.format(m1_diffList06))
    print('With LC = 0.7: {}'.format(m1_diffList07))
    print('With LC = 0.8: {}'.format(m1_diffList08))
    print('With LC = 0.9: {}'.format(m1_diffList09))
    print('With LC = 1.0: {}'.format(m1_diffList10))
    print('')
    print('With LC = 0.0: {} profiles with beneficial manipulations.'.format(m1_numMan00))
    print('With LC = 0.1: {} profiles with beneficial manipulations.'.format(m1_numMan01))
    print('With LC = 0.2: {} profiles with beneficial manipulations.'.format(m1_numMan02))
    print('With LC = 0.3: {} profiles with beneficial manipulations.'.format(m1_numMan03))
    print('With LC = 0.4: {} profiles with beneficial manipulations.'.format(m1_numMan04))
    print('With LC = 0.5: {} profiles with beneficial manipulations.'.format(m1_numMan05))
    print('With LC = 0.6: {} profiles with beneficial manipulations.'.format(m1_numMan06))
    print('With LC = 0.7: {} profiles with beneficial manipulations.'.format(m1_numMan07))
    print('With LC = 0.8: {} profiles with beneficial manipulations.'.format(m1_numMan08))
    print('With LC = 0.9: {} profiles with beneficial manipulations.'.format(m1_numMan09))
    print('With LC = 1.0: {} profiles with beneficial manipulations.'.format(m1_numMan10))

    print('')
    print('*********** MODEL 2 ************')
    print('With LC = 0.0: {}'.format(m2_diffList00))
    print('With LC = 0.1: {}'.format(m2_diffList01))
    print('With LC = 0.2: {}'.format(m2_diffList02))
    print('With LC = 0.3: {}'.format(m2_diffList03))
    print('With LC = 0.4: {}'.format(m2_diffList04))
    print('With LC = 0.5: {}'.format(m2_diffList05))
    print('With LC = 0.6: {}'.format(m2_diffList06))
    print('With LC = 0.7: {}'.format(m2_diffList07))
    print('With LC = 0.8: {}'.format(m2_diffList08))
    print('With LC = 0.9: {}'.format(m2_diffList09))
    print('With LC = 1.0: {}'.format(m2_diffList10))
    print('')
    print('With LC = 0.0: {} profiles with beneficial manipulations.'.format(m2_numMan00))
    print('With LC = 0.1: {} profiles with beneficial manipulations.'.format(m2_numMan01))
    print('With LC = 0.2: {} profiles with beneficial manipulations.'.format(m2_numMan02))
    print('With LC = 0.3: {} profiles with beneficial manipulations.'.format(m2_numMan03))
    print('With LC = 0.4: {} profiles with beneficial manipulations.'.format(m2_numMan04))
    print('With LC = 0.5: {} profiles with beneficial manipulations.'.format(m2_numMan05))
    print('With LC = 0.6: {} profiles with beneficial manipulations.'.format(m2_numMan06))
    print('With LC = 0.7: {} profiles with beneficial manipulations.'.format(m2_numMan07))
    print('With LC = 0.8: {} profiles with beneficial manipulations.'.format(m2_numMan08))
    print('With LC = 0.9: {} profiles with beneficial manipulations.'.format(m2_numMan09))
    print('With LC = 1.0: {} profiles with beneficial manipulations.'.format(m2_numMan10))


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

def print_info(nr, Name, M, myPref):


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
    simulationModel1(9, 3, 500)

def test4_simulationModel1vs2():
    simulationModel1vs2(9, 3, 500)

## Choose TEST

#test2_bestManipulation()
test2_casesHighestDiff()
#test3_bestManipulationFor14()
#test3_bestManipulationFor25()
#test3_bestManipulationFor36()
#test4_simulationModel1vs2()