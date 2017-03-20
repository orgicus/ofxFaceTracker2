#pragma once

#include "ofMain.h"
#include "ofxFaceTracker2.h"
#include "ofxBiquadFilter.h"

typedef dlib::matrix<double,40,1> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;

typedef dlib::decision_function<kernel_type> dec_funct_type;
typedef dlib::normalized_function<dec_funct_type> funct_type;

typedef dlib::probabilistic_decision_function<kernel_type> probabilistic_funct_type;
typedef dlib::normalized_function<probabilistic_funct_type> pfunct_type;


class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    
    sample_type makeSample();
    
    ofxFaceTracker2 tracker;
    ofVideoGrabber grabber;
    
    ofxBiquadFilter1f predictionEaseFilter;
    
    funct_type learned_function;
    
    //training
    std::vector<sample_type> samples;
    std::vector<double> labels;
    dlib::vector_normalizer<sample_type> normalizer;
    
    void addPositiveSample();
    void addNegativeSample();
    void train();

};
