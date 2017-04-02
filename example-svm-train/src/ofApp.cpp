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
    
    ofDrawBitmapString("Usage:\n\nPress 'p' to add a positive sample\nPress 'n' to add a negative sample\nPress 't' to train",10,ofGetHeight()-100);

    ofPushMatrix();
    ofTranslate(0, 100);
    string str = "YOUR MOUTH EXPRESSION";
    float val = predictionEaseFilter.value();
    
    ofDrawBitmapStringHighlight(str, 20, 0);
    ofDrawRectangle(20, 20, 300*val, 30);
    
    ofNoFill();
    ofDrawRectangle(20, 20, 300, 30);
    ofFill();

    ofPopMatrix();

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
    }
}
void ofApp::addNegativeSample(){
    if(tracker.size() > 0){
        sample_type samp = makeSample();
        samples.push_back(samp);
        labels.push_back(-1);
    }
}
//most of the code is "kindly borrowed" directly from dlib's svm_ex.cpp example
void ofApp::train(){
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);
    // now normalize each sample
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);
    
    randomize_samples(samples, labels);
    // The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1
    // labels in the training data.  This function finds that value.
    double max_nu = dlib::maximum_nu(labels);//maximum_nu(labels);
    
    dlib::svm_nu_trainer<kernel_type> trainer;
    
    // Now we loop over some different nu and gamma values to see how good they are.  Note
    // that this is a very simple way to try out a few possible parameter choices.  You
    // should look at the model_selection_ex.cpp program for examples of more sophisticated
    // strategies for determining good parameter choices.
    cout << "doing cross validation" << endl;
    for (double gamma = 0.00001; gamma <= 1; gamma *= 5)
    {
        for (double nu = 0.00001; nu < max_nu; nu *= 5)
        {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_nu(nu);
            
            cout << "gamma: " << gamma << "    nu: " << nu;
            // Print out the cross validation accuracy for 3-fold cross validation using
            // the current gamma and nu.  cross_validate_trainer() returns a row vector.
            // The first element of the vector is the fraction of +1 training examples
            // correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.
            cout << "     cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);
        }
    }
    //        gamma: 0.03125    nu: 0.00025     cross validation accuracy: 1 1
    //        gamma: 0.00625    nu: 0.00125     cross validation accuracy: 1 1
    
    // From looking at the output of the above loop it turns out that a good value for nu
    // and gamma for this problem is 0.15625 for both.  So that is what we will use.
    
    // Now we train on the full set of data and obtain the resulting decision function.  We
    // use the value of 0.15625 for nu and gamma.  The decision function will return values
    // >= 0 for samples it predicts are in the +1 class and numbers < 0 for samples it
    // predicts to be in the -1 class.
    trainer.set_kernel(kernel_type(0.00625));
    trainer.set_nu(0.00125);
    
    // Here we are making an instance of the normalized_function object.  This object
    // provides a convenient way to store the vector normalization information along with
    // the decision function we are going to learn.
    learned_function.normalizer = normalizer;  // save normalization information
    learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results
    
    pfunct_type learned_pfunct;
    learned_pfunct.normalizer = normalizer;
    learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
    
    dlib::serialize(ofToDataPath("data_your_expression.func")) << learned_pfunct;
    
    cout << "saved " << ofToDataPath("data_your_expression.func") << endl;
}

void ofApp::keyPressed(int key){
    if(key == 'p'){
        addPositiveSample();
    }
    if(key == 'n'){
        addNegativeSample();
    }
    if(key == 't'){
        train();
    }
}
