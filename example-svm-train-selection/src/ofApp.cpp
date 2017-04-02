#include <typeinfo>
#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    // Setup value filters for the classifer
    predictionEaseFilter.setFc(0.04);

    // All examples share data files from example-data, so setting data path to this folder
    // This is only relevant for the example apps
    ofSetDataPathRoot(ofFile(__BASE_FILE__).getEnclosingDirectory()+"../../model/");
    
    // Setup grabber
    grabber.setup(1280,720);
    
    // Setup tracker
    tracker.setup();
    
}

//--------------------------------------------------------------
void ofApp::update(){
    grabber.update();
    if(grabber.isFrameNew()){
        tracker.update(grabber);
        
        if(tracker.size() > 0){
            // Run the classifiers and update the filter
            predictionEaseFilter.update(learned_function(makeSample()));
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    grabber.draw(0, 0);
    tracker.drawDebug();
    
#ifndef __OPTIMIZE__
    ofSetColor(ofColor::red);
    ofDrawBitmapString("Warning! Run this app in release mode to get proper performance!",10,60);
    ofSetColor(ofColor::white);
#endif
    
    
    ofPushMatrix();
    ofTranslate(0, 100);
    string str = "YOUR EXPRESSION";
    float val = predictionEaseFilter.value();
    
    ofDrawBitmapStringHighlight(str, 20, 0);
    ofDrawRectangle(20, 20, 300*val, 30);
    
    ofNoFill();
    ofDrawRectangle(20, 20, 300, 30);
    ofFill();

    ofPopMatrix();
    
    drawCrossValidationResults();
    //draw usage
    ofDrawBitmapString("Usage:\n\nPress 'p' to add a positive sample (currently "+ofToString(numPositive)+")\nPress 'n' to add a negative sample (currently "+ofToString(numNegative)+")\nPress 'v' for cross validation\nPress '1' to '0' to select cross validation gamma,nu values\nPress 't' to train\nPress 's' to save data",10,ofGetHeight()-100);
}

void ofApp::drawCrossValidationResults(){
    ofPushMatrix();
    ofTranslate(20, 170);
    for(int i = 0 ; i < topCrossValidationResults.size(); i++){
        cross_validation_result r = topCrossValidationResults[i];
        int key = (i+1);
        if(key == 10) key = 0;
        if(i == selectedValidationResult){
            ofSetColor(255,0,0);
        }else{
            ofSetColor(255);
        }
        ofDrawBitmapString("gamma: " + ofToString(r.gamma) + " , nu: " + ofToString(r.nu) + " cross validation: " + ofToString(r.result(0)) + " , " + ofToString(r.result(0)) + " (key "+ofToString(key)+")",0,20 * i);
        
    }
    ofPopMatrix();
    ofSetColor(255);
}

// Function that creates a sample for the classifier containing the mouth and eyes
sample_type ofApp::makeSample(){
    auto outer = tracker.getInstances()[0].getLandmarks().getImageFeature(ofxFaceTracker2Landmarks::OUTER_MOUTH);
    auto inner = tracker.getInstances()[0].getLandmarks().getImageFeature(ofxFaceTracker2Landmarks::INNER_MOUTH);
    
    auto lEye = tracker.getInstances()[0].getLandmarks().getImageFeature(ofxFaceTracker2Landmarks::LEFT_EYE);
    auto rEye = tracker.getInstances()[0].getLandmarks().getImageFeature(ofxFaceTracker2Landmarks::RIGHT_EYE);
    
    ofVec2f vec = rEye.getCentroid2D() - lEye.getCentroid2D();
    float rot = vec.angle(ofVec2f(1,0));
    
    vector<ofVec2f> relativeMouthPoints;
    
    ofVec2f centroid = outer.getCentroid2D();
    for(ofVec2f p : outer.getVertices()){
        p -= centroid;
        p.rotate(rot);
        p /= vec.length();
        
        relativeMouthPoints.push_back(p);
    }
    
    for(ofVec2f p : inner.getVertices()){
        p -= centroid;
        p.rotate(rot);
        p /= vec.length();
        
        relativeMouthPoints.push_back(p);
    }
    
    sample_type s;
    for(int i=0;i<20;i++){
        s(i*2+0) = relativeMouthPoints[i].x;
        s(i*2+1) = relativeMouthPoints[i].y;
    }
    return s;
}

void ofApp::addPositiveSample(){
    if(tracker.size() > 0){
        sample_type samp = makeSample();
        samples.push_back(samp);
        labels.push_back(+1);
        numPositive++;
    }else{
        ofLogWarning("addPositiveSample","no faces tracked yet, can't add sample");
    }
}
void ofApp::addNegativeSample(){
    if(tracker.size() > 0){
        sample_type samp = makeSample();
        samples.push_back(samp);
        labels.push_back(-1);
        numNegative++;
    }else{
        ofLogWarning("addNegativeSample","no faces tracked yet, can't add sample");
    }
}
void ofApp::clearSamples(){
    numNegative = numPositive = 0;
    samples.clear();
    labels.clear();
    topCrossValidationResults.clear();
}
//most of the code is "kindly borrowed" directly from dlib's svm_ex.cpp example
void ofApp::crossValidate(){
    topCrossValidationResults.clear();
    
    // let the normalizer learn the mean and standard deviation of the samplesx¤
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);
    
    randomize_samples(samples, labels);
    // The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1
    // labels in the training data.  This function finds that value.
    double max_nu = dlib::maximum_nu(labels);//maximum_nu(labels);
    
    // Now we loop over some different nu and gamma values to see how good they are.  Note
    // that this is a very simple way to try out a few possible parameter choices.  You
    // should look at the model_selection_ex.cpp program for examples of more sophisticated
    // strategies for determining good parameter choices.
    ofLogVerbose("crossValidate()", "doing cross validation");
    
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double nu = 0.00001; nu < max_nu; nu *= 5)
        {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_nu(nu);
            
            // Print out the cross validation accuracy for 3-fold cross validation using
            // the current gamma and nu.  cross_validate_trainer() returns a row vector.
            // The first element of the vector is the fraction of +1 training examples
            // correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.
            
            dlib::matrix<double> result = cross_validate_trainer(trainer, samples, labels, 3);

            ofLogVerbose("crossValidate()","gamma: " + ofToString(gamma) + "  nu: " + ofToString(nu) +  "  cross validation accuracy: " + ofToString(result(0)) + " , " + ofToString(result(1)));
            
            if(!isnan(result(0)) && !isnan(result(1))) {
            
                cross_validation_result preset;
                preset.result = result;
                preset.gamma = gamma;
                preset.nu = nu;
                
                topCrossValidationResults.push_back(preset);
                
            }else{
                ofLogWarning("cross validation", "cross validation returning nan, probably, you shuold probably add more samples");
            }
            
        }
    }
    
    // sort cross validation results by highest cross validation accuracy
    sort(topCrossValidationResults.begin(), topCrossValidationResults.end(), compare_result());
    
    //trim top 10 results based on highest cross validation accuracy values
    if(topCrossValidationResults.size() > 10){
        topCrossValidationResults.resize(TOP_SIZE);
    }
    
}
void ofApp::train(){
    if(topCrossValidationResults.size() > 0){
        // Now we train on the full set of data and obtain the resulting decision function.  We
        // use the selected values for nu and gamma.  The decision function will return values
        // >= 0 for samples it predicts are in the +1 class and numbers < 0 for samples it
        // predicts to be in the -1 class.
        double gamma = topCrossValidationResults[selectedValidationResult].gamma;
        double nu = topCrossValidationResults[selectedValidationResult].nu;
        trainer.set_kernel(kernel_type(gamma));
        trainer.set_nu(nu);
        
        // Here we are making an instance of the normalized_function object.  This object
        // provides a convenient way to store the vector normalization information along with
        // the decision function we are going to learn.
        learned_function.normalizer = normalizer;  // save normalization information
        learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results
        
        learned_pfunct.normalizer = normalizer;
        learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
    }else{
        ofLogError("train()", "error training! make sure you have samples and they're cross validated :)");
    }
}

void ofApp::save(){
    if(topCrossValidationResults.size() > 0){
        string path = ofToDataPath("data_your_expression.func");
        dlib::serialize(path) << learned_pfunct;
        ofLogVerbose("save()","saved "+ path);
    }else{
        ofLogError("save()", "please cross validate and train first");
    }
}

void ofApp::keyPressed(int key){
    if(key == 'p'){
        addPositiveSample();
    }
    if(key == 'n'){
        addNegativeSample();
    }
    if(key == 'c'){
        clearSamples();
    }
    if(key == 'v'){
        crossValidate();
    }
    if(key == 't'){
        train();
    }
    if(key == 's'){
        save();
    }
    if(key == '1' && topCrossValidationResults.size() > 0){
        selectedValidationResult = 0;
    }
    if(key == '2' && topCrossValidationResults.size() > 1){
        selectedValidationResult = 1;
    }
    if(key == '3' && topCrossValidationResults.size() > 2){
        selectedValidationResult = 2;
    }
    if(key == '4' && topCrossValidationResults.size() > 3){
        selectedValidationResult = 3;
    }
    if(key == '5' && topCrossValidationResults.size() > 4){
        selectedValidationResult = 4;
    }
    if(key == '6' && topCrossValidationResults.size() > 5){
        selectedValidationResult = 5;
    }
    if(key == '7' && topCrossValidationResults.size() > 6){
        selectedValidationResult = 6;
    }
    if(key == '8' && topCrossValidationResults.size() > 7){
        selectedValidationResult = 7;
    }
    if(key == '9' && topCrossValidationResults.size() > 8){
        selectedValidationResult = 8;
    }
    if(key == '0' && topCrossValidationResults.size() > 9){
        selectedValidationResult = 9;
    }
}
