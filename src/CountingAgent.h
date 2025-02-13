// *******************************
// CTRNN
// *******************************

#pragma once

#include "CTRNN.h"

// The CountingAgent class declaration

class CountingAgent {
	public:
		// The constructor
		CountingAgent(int networksize)
		{
			Set(networksize);
		};
		// The destructor
		~CountingAgent() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetFoodSensorWeight(int to, double value) {foodsensorweights[to] = value;};
		void SetLandmarkSensorWeight(int to, double value) {landmarksensorweights[to] = value;};
		void SetOtherSensorWeight(int to, double value) {othersensorweights[to] = value;};
		void SetFoodSensorState(double state) {foodSensor = state;};
		void SetLandmarkSensorState(double state) {landmarkSensor = state;};

		double FoodSensorWeight(int to) {return foodsensorweights[to];};
		double LandmarkSensorWeight(int to) {return landmarksensorweights[to];};
		double OtherSensorWeight(int to) {return othersensorweights[to];};

		// Control
        void Set(int networksize);
		void ResetPosition(double initpos);
		void ResetNeuralState();
		void SenseLandmarks(double ref, double sep, int numlandmarks, int foodpos);
		void SenseOther(double pos);
		void Step(double StepSize);

		int size;
		double pos, gain, foodSensor, landmarkSensor, otherSensor;
		TVector<double> foodsensorweights, landmarksensorweights, othersensorweights;
		CTRNN NervousSystem;
};
