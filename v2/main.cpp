#include <iostream>
#include "TSearch.h"
#include "CountingAgent.h"
#include "CTRNN.h"
#include "random.h"

// #define PRINTOFILE

// Task params
const int LN = 2;                   // Number of landmarks in the environment
const int VAR = 3;                  // Number of variations to the landmarks
const double StepSize = 0.1;
const double RunDuration = 300.0;
const double TransDuration = 150.0;
const double MinLength = 50.0;      
const double mindist = 5.0;         

// EA params
const int POPSIZE = 96;
const int GENS = 10000;
const double MUTVAR = 0.05;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 3;
const double WR = 10.0;     
const double SR = 10.0;     
const double BR = 10.0;     
const double TMIN = 1.0;
const double TMAX = 16.0;   

// Genotype size
int VectSize = 2 * (N*N + 5*N);  // Double the amount of parameters, one for Receiver, one for Signaller

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
    int k = 1;

    // ------------------------------------------------------------------------
    // Map SIGNALLER params
    // Time-constants
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
        k++;
    }
    // Bias
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -BR, BR);
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }
    // Other Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }    
    
    // ------------------------------------------------------------------------
    // Map RECEIVER params
    // Time-constants
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
        k++;
    }
    // Bias
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -BR, BR);
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
            k++;
        }
    }
    // Food Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }
    // Landmark Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }
    // Other Sensor Weights
    for (int i = 1; i <= N; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
        k++;
    }        
}

// ------------------------------------
// Fitness function
// Landmarks vary position by small amount
// ------------------------------------
double FitnessFunction(TVector<double> &genotype, RandomState &rs)
{
    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    CountingAgent AgentSignaller( N, phenotype, 1);
    CountingAgent AgentReceiver( N, phenotype, (int)(N*N + 5*N + 1));

    // Save state
    TVector<double> savedstateR, savedstateS;
    savedstateR.SetBounds(1,N);
    savedstateS.SetBounds(1,N);

    // Keep track of performance
    float totaltrials = 0;
    double totaltime;
    double distR, distS;
    double totaldistR, totaldistS;
    double totalfitR = 0.0, totalfitS = 0.0;
    double food_loc, food_loc_varied;
    double fitR, fitS;
    
    // Landmarks and variations
    TVector<double> landmarkPositions;
    TVector<double> landmarkPositionsVaried;
    landmarkPositions.SetBounds(1,LN);  // [30, 45, 60..]
    landmarkPositionsVaried.SetBounds(1,LN); 
    for (int i = 1; i <= LN; i += 1)
    {
        landmarkPositions[i] = 15 + i * 15;
    }
    TMatrix<double> landmarkPositionsPossible;
    int variations = (int) pow(VAR,LN);
    landmarkPositionsPossible.SetBounds(1,variations,1,LN);  // [30, 45, 60..]
    double x, y=0;
    double offset = 1.0;
    for (int i = 0; i < variations; i += 1)
    {
        x = i % VAR;
        if (x == 0){ y += 1; }
        landmarkPositionsPossible[i+1][1] = landmarkPositions[1] + x*offset;
        landmarkPositionsPossible[i+1][2] = landmarkPositions[2] + y*offset;
    }

    // Use this to save the neural state during learning
    for (int env = 1; env <= LN; env += 1)
    {
        // Establish food location
        food_loc = landmarkPositions[env];

        // 1. FORAGING PHASE
        AgentSignaller.ResetPosition(0);
        AgentSignaller.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaller.SenseFood(food_loc);
            AgentSignaller.SenseLandmarks(LN,landmarkPositions);
            AgentSignaller.Step(StepSize);
        }
        AgentSignaller.ResetSensors();

        // 2. RECRUITMENT PHASE
        AgentSignaller.ResetPosition(0);
        AgentReceiver.ResetPosition(0);
        AgentReceiver.ResetNeuralState();
        for (double time = 0; time < RunDuration; time += StepSize)
        {
            AgentSignaller.SenseOther(AgentReceiver.pos);
            AgentReceiver.SenseOther(AgentSignaller.pos);
            AgentSignaller.Step(StepSize);
            AgentReceiver.Step(StepSize);
        }
        AgentReceiver.ResetSensors();
        AgentSignaller.ResetSensors();

        // Saved each of their neural states 
        for (int i = 1; i <= N; i++)
        {
            savedstateR[i] = AgentReceiver.NervousSystem.NeuronState(i);
            savedstateS[i] = AgentSignaller.NervousSystem.NeuronState(i);
        }

        // TEST USING DIFFERENT DISTANCES BETWEEN LANDMARKS
        for (int v = 1; v <= variations; v++)
        {
            for (int i = 1; i <= LN; i += 1)
            {
                landmarkPositionsVaried[i] = landmarkPositionsPossible[v][i];
            }
            // Establish food location
            food_loc_varied = landmarkPositionsVaried[env];

            // 3. TESTING PHASE
            AgentReceiver.ResetPosition(0);
            AgentSignaller.ResetPosition(0);
            AgentReceiver.ResetSensors();
            AgentSignaller.ResetSensors();
            // Reset neural state
            for (int i = 1; i <= N; i++)
            {
                AgentReceiver.NervousSystem.SetNeuronState(i, savedstateR[i]);
                AgentSignaller.NervousSystem.SetNeuronState(i, savedstateS[i]);
            }
            
            totaldistR = 0.0; totaldistS = 0.0;
            totaltime = 0.0;
            
            for (double time = 0; time < RunDuration; time += StepSize)
            {
                AgentReceiver.SenseLandmarks(LN,landmarkPositionsVaried); 
                AgentSignaller.SenseLandmarks(LN,landmarkPositionsVaried);
                AgentReceiver.Step(StepSize);
                AgentSignaller.Step(StepSize);

                // Measure distance between them (after transients)
                if (time > TransDuration)
                {
                    distR = fabs(AgentReceiver.pos - food_loc_varied);
                    if (distR < mindist){
                        distR = 0.0;
                    }
                    totaldistR += distR;

                    distS = fabs(AgentSignaller.pos - food_loc_varied);
                    
                    if (distS < mindist){
                        distS = 0.0;
                    }
                    totaldistS += distS;

                    totaltime += 1;
                }
            }
            
            fitR = 1 - ((totaldistR / totaltime)/MinLength);
            if (fitR < 0){
                fitR = 0;
            }
            totalfitR += fitR;

            fitS = 1 - ((totaldistS / totaltime)/MinLength);
            if (fitS < 0.0){
                fitS = 0.0;
            }
            totalfitS += fitS;

            totaltrials += 1;
        }
    }
    return (totalfitR + totalfitS) / (2 * totaltrials);
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
// int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
// {
//     if (BestPerf > 0.99) {
//         return 1;
//     }
//     else {
//         return 0;
//     }
// }

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
    cout << Generation << " " << BestPerf << " " << AvgPerf << endl;
}

void ResultsDisplay(TSearch &s)
{
    std::string current_run = s.CurrentRun();
    std::string dir = s.Directory();

    TVector<double> bestVector;
    ofstream BestIndividualFile;
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);

    // Save the genotype of the best individual
    bestVector = s.BestIndividual();
    BestIndividualFile.open( dir + "best_gen_" + current_run + ".dat");
    BestIndividualFile << bestVector << endl;
    BestIndividualFile.close();

    int k = 1;
    GenPhenMapping(bestVector, phenotype);

    // Show the Signaller
    BestIndividualFile.open( dir + "best_ns_s_" + current_run + ".dat" );
    CountingAgent AgentSignaller( N, phenotype, 1);

    // Send to file
    BestIndividualFile << AgentSignaller.NervousSystem << endl;
    BestIndividualFile << AgentSignaller.foodsensorweights << "\n" << endl;
    BestIndividualFile << AgentSignaller.landmarksensorweights << "\n" << endl;
    BestIndividualFile << AgentSignaller.othersensorweights << "\n" << endl;
    BestIndividualFile.close();

    // Show the Signaller
    BestIndividualFile.open(dir + "best_ns_r_" + current_run + ".dat");
    CountingAgent AgentReceiver( N, phenotype, (int)(N*N + 5*N + 1));

    // Send to file
    BestIndividualFile << AgentReceiver.NervousSystem << endl;
    BestIndividualFile << AgentReceiver.foodsensorweights << "\n" << endl;
    BestIndividualFile << AgentReceiver.landmarksensorweights << "\n" << endl;
    BestIndividualFile << AgentReceiver.othersensorweights << "\n" << endl;
    BestIndividualFile.close();

}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[])
{
    long randomseed = static_cast<long>(time(NULL));
    randomseed += atoi(argv[1]);
    std::string current_run = argv[1];
    std::string dir = "/Users/edizquie/Documents/GitHub/BeeCommunication/E1/";

    TSearch s(VectSize);

    #ifdef PRINTOFILE

    ofstream file;
    file.open  (dir + "evol_" + current_run + ".dat");
    cout.rdbuf(file.rdbuf());

    // save the seed to a file
    ofstream seedfile;
    seedfile.open (dir + "seed_" + current_run + ".dat");
    seedfile << randomseed << endl;
    seedfile.close();
    
    #endif
    
    // Configure the search
    s.SetRandomSeed(randomseed);
    s.SetDir(dir);
    s.SetCurrentRun(current_run);
    s.SetSearchResultsDisplayFunction(ResultsDisplay);
    s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    s.SetSelectionMode(RANK_BASED);
    s.SetReproductionMode(GENETIC_ALGORITHM);
    s.SetPopulationSize(POPSIZE);
    s.SetMaxGenerations(GENS);
    s.SetCrossoverProbability(CROSSPROB);
    s.SetCrossoverMode(UNIFORM);
    s.SetMutationVariance(MUTVAR);
    s.SetMaxExpectedOffspring(EXPECTED);
    s.SetElitistFraction(ELITISM);
    s.SetSearchConstraint(1);

    /* Initialize and seed the search */
    s.InitializeSearch();
    
    /* Evolve */
    //s.SetSearchTerminationFunction(TerminationFunction);
    s.SetEvaluationFunction(FitnessFunction);
    s.ExecuteSearch();

    #ifdef PRINTTOFILE
        evolfile.close();
    #endif

    return 0;
}
