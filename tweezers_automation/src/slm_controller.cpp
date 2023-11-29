// SLMController.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#pragma comment(lib, "ws2_32.lib")

#include <winsock2.h>
#include <iostream>
#include <fstream>

#include "Serial.h"
#include <string>
#include <windows.h>

#include <WS2tcpip.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <shellapi.h>
#include <format>
#include <thread>
#include <mutex>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "bgapi2_genicam.hpp"
#include "spot.h"



//using namespace std;
//using namespace std::chrono;

#define PORT 8080
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define COUT(string) cout<< string << endl
#define CAM_FRAME_RATE 100

// Global variables
Spot spots[100];
int num_spots = 0;
SOCKET udp_socket;
std::mutex m;
std::mutex k;
cv::Mat cam_img;
std::vector<cv::KeyPoint> keypoints;

// Function declarations
char* read_file(std::string filePath);
void initialize_holo_engine();
int send_message(char* message);
int update_uniform(int uniform_var, float values[], int num_values);
int create_spot(float* spot_data);
int modify_spot(float* spot_data, int spot_index);
void random_spots_test();
float* get_spots();
int get_img();
int detect_beads();


// Inputs: file path to the file being read
// Returns: a character array of the message
char* read_file(std::string filePath) {
    using namespace std::this_thread;
    using namespace std::chrono_literals;
    std::string line;
    std::ifstream myfile(filePath);

    if (myfile.is_open())
    {
        char* msg_char = new char[10001];
        int curr_index = 0;
        while (getline(myfile, line)) {
            for (int i = 0; i < line.length(); i++) {
                msg_char[curr_index] = line[i];
                curr_index += 1;
            }
            msg_char[curr_index] = '\n';
            curr_index += 1;
        }

        char* msg_string = msg_char;
        myfile.close();

        return msg_string;
    }
    return NULL;
}

// Connects to the hologram engine by binding to the udp port
// Sends shader code and initial uniform vars values to hologram engine
void initialize_holo_engine() {
    // Initialize winsock and socket
    WSAData wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in bind_addr;

    // Define ip address and port
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = 61556;
    inet_pton(AF_INET, "127.0.0.1", &bind_addr.sin_addr);

    sockaddr_in send_addr;
    send_addr.sin_family = AF_INET;
    send_addr.sin_port = 61557;
    inet_pton(AF_INET, "127.0.0.1", &send_addr.sin_addr);

    // Bind to the hologram engine
    bind(udp_socket, (sockaddr*)&bind_addr, sizeof(bind_addr));


    /*
    const int BufLen = 10240;
    char RecvBuf[BufLen];

    int bind_addrSize = sizeof(bind_addr);

    int bytesReceived = recvfrom(udp_socket, RecvBuf, BufLen, 0, (sockaddr*)&bind_addr, &bind_addrSize);

    cout << bytesReceived;  // print out the bytes we receive from hologram engine

    if (bytesReceived == SOCKET_ERROR) {
        std::cerr << "recvfrom failed." << std::endl;
    }
    */

    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    // Read shader source and uniform vars into variables
    char* shader_source = read_file("./shader_source.txt");
    char* init_uniform_vars = read_file("./init_uniform_vars.txt");
    size_t shader_source_length = strlen(shader_source);
    size_t init_length = strlen(init_uniform_vars);
    // Send code and variables to hologram engine
    int result_shader = sendto(udp_socket, shader_source, shader_source_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr)); // magic number is bytes of array
    int result_init = sendto(udp_socket, init_uniform_vars, init_length, 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    //if (result_shader < 0 || result_init < 0) {
    //    std::cerr << "shader or init code failed to send" << std::endl;
    //}

    //closesocket(udp_socket);
    //WSACleanup();
}

int send_message(char* message) {
    sockaddr_in send_addr;
    send_addr.sin_family = AF_INET;
    send_addr.sin_port = 61557;
    inet_pton(AF_INET, "127.0.0.1", &send_addr.sin_addr);
    int result = sendto(udp_socket, message, sizeof(message), 0, (sockaddr*)&send_addr, sizeof(send_addr));

    return 0;
}

// Updates the specified uniform variable to desired value
int update_uniform(int uniform_var, float values[], int num_values) {
    sockaddr_in recv_addr;
    unsigned short Port = 61557;
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(Port);
    inet_pton(AF_INET, "127.0.0.1", &recv_addr.sin_addr);

    std::string packet = std::format("<data>\n<uniform id = {}>\n", uniform_var);

    for (int i = 0; i < num_values; i++) {
        packet += std::to_string(values[i]) + " ";
    }
    packet += "\n</uniform>\n</data>";

    int result_init = sendto(udp_socket, &packet[0], size(packet), 0, (SOCKADDR*)&recv_addr, sizeof(recv_addr));

    return result_init;
}

// Returns pointer to an array holding all the spot parameters
float* get_spots() {
    float* spot_vals = new float[num_spots * 16];
    int count = 0;
    // for every spot in the spot array, add the parameters into this array
    for (int i = 0; i < num_spots; i++) {
        for (int j = 0; j < 16; j++) {
            spot_vals[count] = spots[i].vals[j];
            count += 1;
        }
    }

    return spot_vals;
}

// Create a new spot and send over parameters to the hologram engine
// Inputs: float array of spot data/parameters
// Returns: number of bytes sent to hologram engine
int create_spot(float* spot_data) {
    // Todo: implement a check for bounds in physical space (120x90um)
    Spot* new_spot = new Spot(spot_data);
    num_spots += 1;
    if (num_spots == 0) {
        spots[0] = *new_spot;
    }
    else {
        spots[num_spots] = *new_spot;

    }

    float* spot_vals = get_spots();
    int update_code = update_uniform(2, spot_vals, sizeof(float) * num_spots * 4);
    return update_code;
}

// Modifies the specified spot and sends updated uniform variable to hologram engine
// Inputs: Spot index, float array of spot data/parameters
// Returns: number of bytes sent to hologram engine
int modify_spot(float* spot_data, int spot_index) {
    spots[spot_index].set_spot_values(spot_data);
    float* spot_vals = get_spots();
    int update_code = update_uniform(2, spot_vals, sizeof(float) * num_spots * 4);
    return update_code;
}

// Randomly creates and modifies 50 spots to test the connection and hologram engine
void random_spots_test() {
    srand(static_cast<unsigned>(time(0)));

    // Create 50 random spots
    for (int i = 0; i < 50; i++) {
        float random_spot_data[16];
        for (int j = 0; j < 16; j++) {
            random_spot_data[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0;
        }

        create_spot(random_spot_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Modify 50 random spots
    for (int i = 0; i < 50; i++) {
        int spot_index = rand() % num_spots;
        float random_spot_data[16];
        for (int j = 0; j < 16; j++) {
            random_spot_data[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0;
        }

        modify_spot(random_spot_data, spot_index);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Creates a trap at a specified point and moves it horizontally at a specified velocity
// Inputs: x and y are initial coords of trap (in micrometers (um)), um_sec um/sec 
// um_distance movement distance
void line_path(float y, float x, int um_sec, int um_distance) {
    int slm_refresh = 100; // refresh rate of slm (fastest we can update the hologram)
    float spot_params[16] = { y, -x, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0,
                            0.0, 0.0, 0.0, 0.0 };
    create_spot(spot_params);
    // pxls / second
    // for loop runs 10 times per second
    int time = 0;
    int pxl = x;
    while (true) {
        // element 0 x  y  z  l    (x,y,z in um and l is an integer)
        // element 1 intensity (I) phase -  -
        // element 2 na.x na.y na.r -  (the x, y, and radius, of the spot on the SLM- useful for Shack-Hartmann holograms)
        // element 3 line trapping x y z and phase gradient.  xyz define the size and angle of the line, phase gradient (between +/-1) is the
        // scattering force component along the line.  Zero is usually a good choice for in-plane line traps
        float n_spot_params[16] = { y, -pxl, 0.0, 0.0,
                                    1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0 };
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / um_sec));
        modify_spot(n_spot_params, 0);
        pxl += 1.0;
        //COUT("pxl_move");
        if (pxl == um_distance + x) {
            break;
        }
    }

}

void testing_line_path() {
    initialize_holo_engine(); // bind to the udp socket and intialize shader code

    line_path(70.0, 0.0, 10, 120);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    line_path(0.0, 0.0, 10, 20);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    line_path(35.0, 0.0, 10, 120);
}

void test_serial() {
    tstring commPortName(TEXT("COM3"));
    Serial serial(commPortName, 57600);
    char set_pow[] = "SDC 35";
    int bytesWritten = serial.write(set_pow);
    std::cout << std::format("{} bytes written to serial port", bytesWritten) << std::endl;
    char buffer[20];

    std::cout << "Reading from the serial port: ";
    for (int i = 0; i < 10; i++)
    {
        int charsRead = serial.read(buffer, 20);
        std::cout << buffer;
        Sleep(100);
    }
    std::cout << std::endl;

}

void test_cam_detect() {
    k.lock();
    for (int i = 0; i < keypoints.size(); i++) {
        std::cout << "start trapping all beads";
        float spot_params[16] = { keypoints[i].pt.y * 0.1875, -keypoints[i].pt.x * 0.1875, 0.0, 0.0,
                                1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0,
                                0.0, 0.0, 0.0, 0.0 };
        create_spot(spot_params);
    }
    k.unlock();

    int move_dist = 100;  // move 100um

    // Move all beads to right edge of screen
    int um_sec = 1;
    for (int i = 0; i < move_dist; i++) {
        //m.lock();
        //detect_beads();
        //m.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / um_sec));
        k.lock();
        for (int j = 0; j < keypoints.size(); j++) {
            if (keypoints[j].pt.x < 640) {
                std::cout << "Keypoints y, x:" << " ";
                std::cout << keypoints[j].pt.y << ", ";
                std::cout << keypoints[j].pt.x << std::endl;

                float n_spot_params[16] = { keypoints[j].pt.y * 0.1875,  (-keypoints[j].pt.x * 0.1875) - 1, 0.0, 0.0,
                                        1.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 1.0,
                                        0.0, 0.0, 0.0, 0.0 };
                modify_spot(n_spot_params, j);
            }
        }
        k.unlock();
    }

}

int main()
{
    initialize_holo_engine();
    //std::thread imaging(get_img);
    //std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    //std::thread detecting(detect_beads);

    //test_cam_detect();
}

int detect_beads() {
    //high_resolution_clock::time_point t0 = high_resolution_clock::now();

    //duration<double> time_span = duration_cast<duration<double>>(t0 - t0);

    //cv::Mat a = ReadMatFromTxt("a.txt", 600, 1024);

    //cv::Mat b = ReadMatFromTxt("b.txt", 600, 1024);
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cv::Mat oriimg;

        oriimg = cv::imread("C:\\Users\\User\\Desktop\\Ekta_code_august2017_dicty\\ConsoleApplication1\\201.jpg", cv::IMREAD_GRAYSCALE); // Read the file
        m.lock();
        //oriimg = cam_img.clone();
        m.unlock();

        if (oriimg.empty()) // Check for invalid input
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        cv::Mat img;
        cv::Mat img1;
        cv::Mat img2;
        cv::Mat edges;
        int h = oriimg.cols; int w = oriimg.rows;

        //cv::subtract(oriimg, b, img1, cv::noArray(), CV_64FC1);
        //cv::divide(img1, a, img2);

        //Step 1: detecting beads


        oriimg.convertTo(img, CV_8UC1);
        cv::Mat Step1;

        int offset = 2;

        //high_resolution_clock::time_point t1 = high_resolution_clock::now();

        cv::equalizeHist(img, Step1);

        cv::medianBlur(Step1, Step1, 3);
        cv::copyMakeBorder(Step1, Step1, offset, offset, offset, offset, cv::BORDER_CONSTANT, cv::Scalar(255));

        cv::SimpleBlobDetector::Params params;
        params.minArea = 300;
        params.minCircularity = 0.85;
        params.filterByArea = 1;
        params.filterByInertia = 1;
        params.filterByCircularity = 1;
        params.filterByColor = 1;

        //std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);
        k.lock();
        blobDetector->detect(Step1, keypoints);
        //high_resolution_clock::time_point t2 = high_resolution_clock::now();
        cv::Mat imgWithKeypoints;

        
        for (int i = 0; i < keypoints.size(); i++) {
            keypoints[i].pt = cv::Point(keypoints[i].pt.x - 2, keypoints[i].pt.y - 2);
            //std::cout << keypoints[i].pt;  // print out coordinates of beads
        }
        //std::cout << std::endl;
        k.unlock();
    }
    
}


int get_img() {
    // DECLARATIONS OF VARIABLES
    BGAPI2::ImageProcessor* imgProcessor = NULL;

    BGAPI2::SystemList* systemList = NULL;
    BGAPI2::System* pSystem = NULL;
    BGAPI2::String sSystemID;

    BGAPI2::InterfaceList* interfaceList = NULL;
    BGAPI2::Interface* pInterface = NULL;
    BGAPI2::String sInterfaceID;

    BGAPI2::DeviceList* deviceList = NULL;
    BGAPI2::Device* pDevice = NULL;
    BGAPI2::String sDeviceID;

    BGAPI2::DataStreamList* datastreamList = NULL;
    BGAPI2::DataStream* pDataStream = NULL;
    BGAPI2::String sDataStreamID;

    BGAPI2::BufferList* bufferList = NULL;
    BGAPI2::Buffer* pBuffer = NULL;
    BGAPI2::String sBufferID;
    int returncode = 0;

    // OPENCV VARIABLE DECLARATIONS
    cv::VideoWriter cvVideoCreator;                 // Create OpenCV video creator
    cv::String videoFileName = "openCvVideo.avi";   // Define video filename
    cv::Size frameSize = cv::Size(2048, 1088);      // Define video frame size (frame width x height)
    cvVideoCreator.open(videoFileName, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, frameSize, true); // set the codec type and frame rate

    // Load image processor
    try {
        imgProcessor = new BGAPI2::ImageProcessor();
        std::cout << "ImageProcessor version:    " << imgProcessor->GetVersion() << std::endl;
        if (imgProcessor->GetNodeList()->GetNodePresent("DemosaicingMethod") == true) {
            // possible values: NearestNeighbor, Bilinear3x3, Baumer5x5
            imgProcessor->GetNodeList()->GetNode("DemosaicingMethod")->SetString("NearestNeighbor");
            std::cout << "    Demosaicing method:    "
                << imgProcessor->GetNodeList()->GetNode("DemosaicingMethod")->GetString() << std::endl;
        }
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "SYSTEM LIST" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // COUNTING AVAILABLE SYSTEMS (TL producers)
    try {
        systemList = BGAPI2::SystemList::GetInstance();
        systemList->Refresh();
        std::cout << "5.1.2   Detected systems:  " << systemList->size() << std::endl;

        // SYSTEM DEVICE INFORMATION
        for (BGAPI2::SystemList::iterator sysIterator = systemList->begin();
            sysIterator != systemList->end();
            sysIterator++) {

            /*std::cout << "  5.2.1   System Name:     " << sysIterator->GetFileName() << std::endl;
            std::cout << "          System Type:     " << sysIterator->GetTLType() << std::endl;
            std::cout << "          System Version:  " << sysIterator->GetVersion() << std::endl;
            std::cout << "          System PathName: " << sysIterator->GetPathName() << std::endl << std::endl;*/
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }


    // OPEN THE FIRST SYSTEM IN THE LIST WITH A CAMERA CONNECTED
    try {
        for (BGAPI2::SystemList::iterator sysIterator = systemList->begin();
            sysIterator != systemList->end();
            sysIterator++) {
            /*std::cout << "SYSTEM" << std::endl;
            std::cout << "######" << std::endl << std::endl;*/

            try {
                sysIterator->Open();
                std::cout << "5.1.3   Open next system " << std::endl;
                std::cout << "  5.2.1   System Name:     " << sysIterator->GetFileName() << std::endl;
                std::cout << "          System Type:     " << sysIterator->GetTLType() << std::endl;
                std::cout << "          System Version:  " << sysIterator->GetVersion() << std::endl;
                std::cout << "          System PathName: " << sysIterator->GetPathName() << std::endl
                    << std::endl;
                sSystemID = sysIterator->GetID();
                /*std::cout << "        Opened system - NodeList Information " << std::endl;
                std::cout << "          GenTL Version:   "
                    << sysIterator->GetNode("GenTLVersionMajor")->GetValue() << "."
                    << sysIterator->GetNode("GenTLVersionMinor")->GetValue() << std::endl << std::endl;*/

                    /* std::cout << "INTERFACE LIST" << std::endl;
                     std::cout << "##############" << std::endl << std::endl;*/

                try {
                    interfaceList = sysIterator->GetInterfaces();
                    // COUNT AVAILABLE INTERFACES
                    interfaceList->Refresh(100);  // timeout of 100 msec
                    /*std::cout << "5.1.4   Detected interfaces: " << interfaceList->size() << std::endl;*/

                    // INTERFACE INFORMATION
                    for (BGAPI2::InterfaceList::iterator ifIterator = interfaceList->begin();
                        ifIterator != interfaceList->end();
                        ifIterator++) {
                        /*std::cout << "  5.2.2   Interface ID:      "
                            << ifIterator->GetID() << std::endl;
                        std::cout << "          Interface Type:    "
                            << ifIterator->GetTLType() << std::endl;
                        std::cout << "          Interface Name:    "
                            << ifIterator->GetDisplayName() << std::endl << std::endl;*/
                    }
                }
                catch (BGAPI2::Exceptions::IException& ex) {
                    returncode = (returncode == 0) ? 1 : returncode;
                    /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
                    std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
                    std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
                }


                /*std::cout << "INTERFACE" << std::endl;
                std::cout << "#########" << std::endl << std::endl;*/

                // OPEN THE NEXT INTERFACE IN THE LIST
                try {
                    for (BGAPI2::InterfaceList::iterator ifIterator = interfaceList->begin();
                        ifIterator != interfaceList->end();
                        ifIterator++) {
                        try {
                            /*std::cout << "5.1.5   Open interface " << std::endl;
                            std::cout << "  5.2.2   Interface ID:      "
                                << ifIterator->GetID() << std::endl;
                            std::cout << "          Interface Type:    "
                                << ifIterator->GetTLType() << std::endl;
                            std::cout << "          Interface Name:    "
                                << ifIterator->GetDisplayName() << std::endl;*/
                            ifIterator->Open();
                            // search for any camera is connetced to this interface
                            deviceList = ifIterator->GetDevices();
                            deviceList->Refresh(100);
                            if (deviceList->size() == 0) {
                                /* std::cout << "5.1.13   Close interface (" << deviceList->size() << " cameras found) "
                                     << std::endl << std::endl;*/
                                ifIterator->Close();
                            }
                            else {
                                sInterfaceID = ifIterator->GetID();
                                /*std::cout << "   " << std::endl;
                                std::cout << "        Opened interface - NodeList Information" << std::endl;*/
                                if (ifIterator->GetTLType() == "GEV") {
                                    /* std::cout << "          GevInterfaceSubnetIPAddress: "
                                         << ifIterator->GetNode("GevInterfaceSubnetIPAddress")->GetValue()
                                         << std::endl;
                                     std::cout << "          GevInterfaceSubnetMask:      "
                                         << ifIterator->GetNode("GevInterfaceSubnetMask")->GetValue()
                                         << std::endl;*/
                                }
                                if (ifIterator->GetTLType() == "U3V") {
                                    // std::cout << "          NodeListCount:     "
                                    // << ifIterator->GetNodeList()->GetNodeCount() << std::endl;
                                }
                                std::cout << "  " << std::endl;
                                break;
                            }
                        }
                        catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                            returncode = (returncode == 0) ? 1 : returncode;
                            /*std::cout << " Interface " << ifIterator->GetID() << " already opened " << std::endl;
                            std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;*/
                        }
                    }
                }
                catch (BGAPI2::Exceptions::IException& ex) {
                    returncode = (returncode == 0) ? 1 : returncode;
                    /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
                    std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
                    std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
                }


                // if a camera is connected to the system interface then leave the system loop
                if (sInterfaceID != "") {
                    break;
                }
            }
            catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                /*std::cout << " System " << sysIterator->GetID() << " already opened " << std::endl;
                std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;*/
            }
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /* std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
         std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
         std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }

    if (sSystemID == "") {
        /* std::cout << " No System found " << std::endl;
         std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";*/
        int endKey = 0;
        std::cin >> endKey;
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pSystem = (*systemList)[sSystemID];
    }


    if (sInterfaceID == "") {
        /*std::cout << " No camera found " << sInterfaceID << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";*/
        int endKey = 0;
        std::cin >> endKey;
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pInterface = (*interfaceList)[sInterfaceID];
    }


    /* std::cout << "DEVICE LIST" << std::endl;
     std::cout << "###########" << std::endl << std::endl;*/

    try {
        // COUNTING AVAILABLE CAMERAS
        deviceList = pInterface->GetDevices();
        deviceList->Refresh(100);
        std::cout << "5.1.6   Detected devices:         " << deviceList->size() << std::endl;

        // DEVICE INFORMATION BEFORE OPENING
        for (BGAPI2::DeviceList::iterator devIterator = deviceList->begin();
            devIterator != deviceList->end();
            devIterator++) {
            /*std::cout << "  5.2.3   Device DeviceID:        "
                << devIterator->GetID() << std::endl;
            std::cout << "          Device Model:           "
                << devIterator->GetModel() << std::endl;
            std::cout << "          Device SerialNumber:    "
                << devIterator->GetSerialNumber() << std::endl;
            std::cout << "          Device Vendor:          "
                << devIterator->GetVendor() << std::endl;
            std::cout << "          Device TLType:          "
                << devIterator->GetTLType() << std::endl;
            std::cout << "          Device AccessStatus:    "
                << devIterator->GetAccessStatus() << std::endl;
            std::cout << "          Device UserID:          "
                << devIterator->GetDisplayName() << std::endl << std::endl;*/
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }


    /*std::cout << "DEVICE" << std::endl;
    std::cout << "######" << std::endl << std::endl;*/

    // OPEN THE FIRST CAMERA IN THE LIST
    try {
        for (BGAPI2::DeviceList::iterator devIterator = deviceList->begin();
            devIterator != deviceList->end();
            devIterator++) {
            try {
                devIterator->Open();
                sDeviceID = devIterator->GetID();
                break;
            }
            catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                std::cout << " Device  " << devIterator->GetID() << " already opened " << std::endl;
                std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;
            }
            catch (BGAPI2::Exceptions::AccessDeniedException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                std::cout << " Device  " << devIterator->GetID() << " already opened " << std::endl;
                std::cout << " AccessDeniedException " << ex.GetErrorDescription() << std::endl;
            }
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    if (sDeviceID == "") {
        std::cout << " No Device found " << sDeviceID << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";
        int endKey = 0;
        std::cin >> endKey;
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pDevice = (*deviceList)[sDeviceID];
    }


    std::cout << "DEVICE PARAMETER SETUP" << std::endl;
    std::cout << "######################" << std::endl << std::endl;

    try {
        // SET TRIGGER MODE OFF (FreeRun)
        pDevice->GetRemoteNode("TriggerMode")->SetString("Off");
        std::cout << "         TriggerMode:             "
            << pDevice->GetRemoteNode("TriggerMode")->GetValue() << std::endl;
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "DATA STREAM LIST" << std::endl;
    std::cout << "################" << std::endl << std::endl;

    try {
        // COUNTING AVAILABLE DATASTREAMS
        datastreamList = pDevice->GetDataStreams();
        datastreamList->Refresh();
        std::cout << "5.1.8   Detected datastreams:     " << datastreamList->size() << std::endl;

        // DATASTREAM INFORMATION BEFORE OPENING
        for (BGAPI2::DataStreamList::iterator dstIterator = datastreamList->begin();
            dstIterator != datastreamList->end();
            dstIterator++) {
            std::cout << "  5.2.4   DataStream ID:          " << dstIterator->GetID() << std::endl << std::endl;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "DATA STREAM" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // OPEN THE FIRST DATASTREAM IN THE LIST
    try {
        if (datastreamList->size() > 0) {
            pDataStream = (*datastreamList)[0];
            std::cout << "5.1.9   Open first datastream " << std::endl;
            std::cout << "          DataStream ID:          " << pDataStream->GetID() << std::endl << std::endl;
            pDataStream->Open();
            sDataStreamID = pDataStream->GetID();
            std::cout << "        Opened datastream - NodeList Information" << std::endl;
            std::cout << "          StreamAnnounceBufferMinimum:  "
                << pDataStream->GetNode("StreamAnnounceBufferMinimum")->GetValue() << std::endl;
            if (pDataStream->GetTLType() == "GEV") {
                std::cout << "          StreamDriverModel:            "
                    << pDataStream->GetNode("StreamDriverModel")->GetValue() << std::endl;
            }
            std::cout << "  " << std::endl;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    if (sDataStreamID == "") {
        std::cout << " No DataStream found" << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";
        int endKey = 0;
        std::cin >> endKey;
        pDevice->Close();
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }

    std::cout << "BUFFER LIST" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    try {
        // BufferList
        bufferList = pDataStream->GetBufferList();

        // 4 buffers using internal buffer mode
        for (int i = 0; i < 4; i++) {
            pBuffer = new BGAPI2::Buffer();
            bufferList->Add(pBuffer);
        }
        std::cout << "5.1.10   Announced buffers:       " << bufferList->GetAnnouncedCount() << " using "
            << pBuffer->GetMemSize() * bufferList->GetAnnouncedCount() << " [bytes]" << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    try {
        for (BGAPI2::BufferList::iterator bufIterator = bufferList->begin();
            bufIterator != bufferList->end();
            bufIterator++) {
            bufIterator->QueueBuffer();
        }
        std::cout << "5.1.11   Queued buffers:          " << bufferList->GetQueuedCount() << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << " " << std::endl;

    std::cout << "CAMERA START" << std::endl;
    std::cout << "############" << std::endl << std::endl;

    // START DataStream acquisition
    try {
        pDataStream->StartAcquisition(4);
        std::cout << "5.1.12   DataStream started " << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // START CAMERA
    try {
        std::cout << "5.1.12   " << pDevice->GetModel() << " started " << std::endl;
        pDevice->GetRemoteNode("AcquisitionStart")->Execute();
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // CAPTURE 8 IMAGES
    //std::cout << " " << std::endl;
    //std::cout << "CAPTURE & TRANSFORM 4 IMAGES" << std::endl;
    //std::cout << "############################" << std::endl << std::endl;

    BGAPI2::Buffer* pBufferFilled = NULL;
    try {
        BGAPI2::Image* pImage = imgProcessor->CreateImage();
        BGAPI2::Node* pPixelFormatInfoSelector = imgProcessor->GetNode("PixelFormatInfoSelector");
        BGAPI2::Node* pBytesPerPixel = imgProcessor->GetNode("BytesPerPixel");

        // create OpenCV window ----
        cv::namedWindow("OpenCV window: Cam");

        while (pDataStream->GetIsGrabbing()) {
            pBufferFilled = pDataStream->GetFilledBuffer(1000);  // timeout 1000 msec
            if (pBufferFilled == NULL) {
                std::cout << "Error: Buffer Timeout after 1000 msec" << std::endl << std::endl;
            }
            else if (pBufferFilled->GetIsIncomplete() == true) {
                //std::cout << "Error: Image is incomplete" << std::endl << std::endl;
                // queue buffer again
                pBufferFilled->QueueBuffer();
            }
            else {
                //std::cout << " Image " << std::setw(5) << pBufferFilled->GetFrameID()
                //   << " received in memory address " << std::hex << pBufferFilled->GetMemPtr()
                //   << std::dec << std::endl;

                // create an image object from the filled buffer and convert it
                BGAPI2::Image* pTransformImage = NULL;
                pImage->Init(pBufferFilled);

                BGAPI2::String sPixelFormat = pImage->GetPixelformat();
                /*
                std::cout << "  pImage.Pixelformat:             "
                    << pImage->GetPixelformat() << std::endl;
                std::cout << "  pImage.Width:                   "
                    << pImage->GetWidth() << std::endl;
                std::cout << "  pImage.Height:                  "
                    << pImage->GetHeight() << std::endl;
                std::cout << "  pImage.Buffer:                  "
                    << std::hex << pImage->GetBuffer() << std::dec << std::endl;
                */
                pPixelFormatInfoSelector->SetValue(sPixelFormat);
                double fBytesPerPixel = pBytesPerPixel->GetAvailable() ? pBytesPerPixel->GetDouble() : 0.0;

                //std::cout << "  Bytes per image:                "
                 //   << static_cast<unsigned int>((pImage->GetWidth()) * (pImage->GetHeight()) * fBytesPerPixel)
                  //  << std::endl;
                //std::cout << "  Bytes per pixel:                "
                 //   << fBytesPerPixel << std::endl;

                // display first 6 pixel values of first 6 lines of the image
                // ========================================================================
                unsigned char* imageBuffer = (unsigned char*)pImage->GetBuffer();

                //std::cout << "  Address" << std::endl;
                // set display for uppercase hex numbers filled with '0'
                //std::cout << std::uppercase << std::setfill('0') << std::hex;
                for (int j = 0; j < 6; j++) {  // first 6 lines
                    void* imageBufferAddress = &imageBuffer[static_cast<int>(pImage->GetWidth() * j * fBytesPerPixel)];
                    //std::cout << "  " << std::setw(8) << imageBufferAddress << " ";
                    for (int k = 0; k < static_cast<int>(6 * fBytesPerPixel); k++) {  // bytes of first 6 pixels
                        //std::cout << " " << std::setw(2)
                            //<< static_cast<int>(imageBuffer[static_cast<int>(pImage->GetWidth() * j * fBytesPerPixel) + k]);
                    }
                    //std::cout << "  ..." << std::endl;
                }
                // set display for lowercase dec numbers filled with ' '
                //std::cout << std::nouppercase << std::setfill(' ') << std::dec;

                // if pixel format starts with "Mono"
                if (std::string(pImage->GetPixelformat()).substr(0, 4) == "Mono") {
                    // transform to Mono8
                    pTransformImage = imgProcessor->CreateTransformedImage(pImage, "Mono8");
                    /*
                    std::cout << " Image "
                        << std::setw(5) << pBufferFilled->GetFrameID() << " transformed to Mono8" << std::endl;
                    std::cout << "  pTransformImage.Pixelformat:    "
                        << pTransformImage->GetPixelformat() << std::endl;
                    std::cout << "  pTransformImage.Width:          "
                        << pTransformImage->GetWidth() << std::endl;
                    std::cout << "  pTransformImage.Height:         "
                        << pTransformImage->GetHeight() << std::endl;
                    std::cout << "  pTransformImage.Buffer:         "
                        << std::hex << pTransformImage->GetBuffer() << std::dec << std::endl;
                    std::cout << "  Bytes per image:                "
                        << pTransformImage->GetWidth() * pTransformImage->GetHeight() * 1 << std::endl;
                    std::cout << "  Bytes per pixel:                "
                        << 1.0 << std::endl;
                    */
                    unsigned char* transformBuffer = (unsigned char*)pTransformImage->GetBuffer();

                    // display first 6 pixel values of first 6 lines of the transformed image
                    // ========================================================================
                    //std::cout << "  Address    Y  Y  Y  Y  Y  Y " << std::endl;

                    // set display for uppercase hex numbers filled with '0'
                    //std::cout << std::uppercase << std::setfill('0') << std::hex;
                    for (int j = 0; j < 6; j++) {  // first 6 lines
                        void* transformBufferAddress = &transformBuffer[pTransformImage->GetWidth() * 1 * j];
                        ///std::cout << "  " << std::setw(8) << std::setfill('0')
                         //   << std::hex << transformBufferAddress << " ";
                        for (int k = 0; k < 6; k++) {  // first 6 Pixel with Mono8 (1 Byte per Pixel)
                            // value of pixel
                           // std::cout << " " << std::setw(2)
                             //   << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * j + k]);
                        }
                        //std::cout << " ..." << std::endl;
                    }
                    // set display for lowercase dec numbers filled with ' '
                    //std::cout << std::nouppercase << std::setfill(' ') << std::dec;
                    //std::cout << " " << std::endl;
                }
                else {  // if color format
                    // transform to BGR8
                    pTransformImage = imgProcessor->CreateTransformedImage(pImage, "BGR8");
                    /*
                    std::cout << " Image "
                        << std::setw(5) << pBufferFilled->GetFrameID() << " transformed to BGR8" << std::endl;
                    std::cout << "  pTransformImage.Pixelformat:    "
                        << pTransformImage->GetPixelformat() << std::endl;
                    std::cout << "  pTransformImage.Width:          "
                        << pTransformImage->GetWidth() << std::endl;
                    std::cout << "  pTransformImage.Height:         "
                        << pTransformImage->GetHeight() << std::endl;
                    std::cout << "  pTransformImage.Buffer:         "
                        << std::hex << pTransformImage->GetBuffer() << std::dec << std::endl;
                    std::cout << "  Bytes per image:                "
                        << pTransformImage->GetWidth() * pTransformImage->GetHeight() * 3 << std::endl;
                    std::cout << "  Bytes per pixel:                "
                        << 3.0 << std::endl;
                    */

                    unsigned char* transformBuffer = (unsigned char*)pTransformImage->GetBuffer();

                    // display first 6 pixel values of first 6 lines of the transformed image
                    // ========================================================================
                    //std::cout << "  Address    B  G  R  B  G  R  B  G  R  B  G  R  B  G  R  B  G  R" << std::endl;

                    // set display for uppercase hex numbers filled with '0'
                    std::cout << std::uppercase << std::setfill('0') << std::hex;
                    for (int j = 0; j < 6; j++) {  // 6 lines
                        void* transformBufferAddress = &transformBuffer[pTransformImage->GetWidth() * 3 * j];
                        std::cout << "  " << std::setw(8) << std::setfill('0')
                            << std::hex << transformBufferAddress << " ";
                        for (int k = 0; k < 6; k++) {  // first 6 Pixel with BGR8 (3 Bytes per Pixel)
                            // Value of Blue pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 0]);
                            // Value of Green pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 1]);
                            // Value of Red pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 2]);
                        }
                        std::cout << " ..." << std::endl;
                    }
                    // set display for lowercase dec numbers filled with ' '
                    std::cout << std::nouppercase << std::setfill(' ') << std::dec;
                    std::cout << " " << std::endl;
                }

                // OPEN CV STUFF
                m.lock();
                cam_img = cv::Mat(pTransformImage->GetHeight(), pTransformImage->GetWidth(), CV_8U, (int*)pTransformImage->GetBuffer());
                m.unlock();
                //display the current image in the window ----
                cv::imshow("Camera", cam_img);
                cv::waitKey(CAM_FRAME_RATE);

                pTransformImage->Release();
                // delete [] transformBuffer;

                // QUEUE BUFFER AFTER USE
                pBufferFilled->QueueBuffer();
            }
        }
        if (pImage != NULL) {
            pImage->Release();
            pImage = NULL;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << " " << std::endl;


    std::cout << "CAMERA STOP" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // STOP CAMERA
    try {
        // SEARCH FOR 'AcquisitionAbort'
        if (pDevice->GetRemoteNodeList()->GetNodePresent("AcquisitionAbort")) {
            pDevice->GetRemoteNode("AcquisitionAbort")->Execute();
            std::cout << "5.1.12   " << pDevice->GetModel() << " aborted " << std::endl;
        }

        pDevice->GetRemoteNode("AcquisitionStop")->Execute();
        std::cout << "5.1.12   " << pDevice->GetModel() << " stopped " << std::endl;
        std::cout << std::endl;

        BGAPI2::String sExposureNodeName = "";
        if (pDevice->GetRemoteNodeList()->GetNodePresent("ExposureTime")) {
            sExposureNodeName = "ExposureTime";
        }
        else if (pDevice->GetRemoteNodeList()->GetNodePresent("ExposureTimeAbs")) {
            sExposureNodeName = "ExposureTimeAbs";
        }
        std::cout << "         ExposureTime:                   "
            << std::fixed << std::setprecision(0) << pDevice->GetRemoteNode(sExposureNodeName)->GetDouble() << " ["
            << pDevice->GetRemoteNode(sExposureNodeName)->GetUnit() << "]" << std::endl;
        if (pDevice->GetTLType() == "GEV") {
            if (pDevice->GetRemoteNodeList()->GetNodePresent("DeviceStreamChannelPacketSize")) {
                std::cout << "         DeviceStreamChannelPacketSize:  "
                    << pDevice->GetRemoteNode("DeviceStreamChannelPacketSize")->GetInt() << " [bytes]" << std::endl;
            }
            else {
                std::cout << "         GevSCPSPacketSize:              "
                    << pDevice->GetRemoteNode("GevSCPSPacketSize")->GetInt() << " [bytes]" << std::endl;
            }
            std::cout << "         GevSCPD (PacketDelay):          "
                << pDevice->GetRemoteNode("GevSCPD")->GetInt() << " [tics]" << std::endl;
        }
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // STOP DataStream acquisition
    try {
        if (pDataStream->GetTLType() == "GEV") {
            // DataStream Statistic
            std::cout << "         DataStream Statistics " << std::endl;
            std::cout << "           DataBlockComplete:              "
                << pDataStream->GetNodeList()->GetNode("DataBlockComplete")->GetInt() << std::endl;
            std::cout << "           DataBlockInComplete:            "
                << pDataStream->GetNodeList()->GetNode("DataBlockInComplete")->GetInt() << std::endl;
            std::cout << "           DataBlockMissing:               "
                << pDataStream->GetNodeList()->GetNode("DataBlockMissing")->GetInt() << std::endl;
            std::cout << "           PacketResendRequestSingle:      "
                << pDataStream->GetNodeList()->GetNode("PacketResendRequestSingle")->GetInt() << std::endl;
            std::cout << "           PacketResendRequestRange:       "
                << pDataStream->GetNodeList()->GetNode("PacketResendRequestRange")->GetInt() << std::endl;
            std::cout << "           PacketResendReceive:            "
                << pDataStream->GetNodeList()->GetNode("PacketResendReceive")->GetInt() << std::endl;
            std::cout << "           DataBlockDroppedBufferUnderrun: "
                << pDataStream->GetNodeList()->GetNode("DataBlockDroppedBufferUnderrun")->GetInt() << std::endl;
            std::cout << "           Bitrate:                        "
                << pDataStream->GetNodeList()->GetNode("Bitrate")->GetDouble() << std::endl;
            std::cout << "           Throughput:                     "
                << pDataStream->GetNodeList()->GetNode("Throughput")->GetDouble() << std::endl;
            std::cout << std::endl;
        }
        if (pDataStream->GetTLType() == "U3V") {
            // DataStream Statistic
            std::cout << "         DataStream Statistics " << std::endl;
            std::cout << "           GoodFrames:            "
                << pDataStream->GetNodeList()->GetNode("GoodFrames")->GetInt() << std::endl;
            std::cout << "           CorruptedFrames:       "
                << pDataStream->GetNodeList()->GetNode("CorruptedFrames")->GetInt() << std::endl;
            std::cout << "           LostFrames:            "
                << pDataStream->GetNodeList()->GetNode("LostFrames")->GetInt() << std::endl;
            std::cout << std::endl;
        }

        pDataStream->StopAcquisition();
        std::cout << "5.1.12   DataStream stopped " << std::endl;
        bufferList->DiscardAllBuffers();
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << std::endl;


    std::cout << "RELEASE" << std::endl;
    std::cout << "#######" << std::endl << std::endl;

    // Release buffers
    std::cout << "5.1.13   Releasing the resources " << std::endl;
    try {
        while (bufferList->size() > 0) {
            pBuffer = *(bufferList->begin());
            bufferList->RevokeBuffer(pBuffer);
            delete pBuffer;
        }
        std::cout << "         buffers after revoke:    " << bufferList->size() << std::endl;

        pDataStream->Close();
        pDevice->Close();
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        std::cout << "         ImageProcessor released" << std::endl << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "End" << std::endl << std::endl;

    std::cout << "Input any number to close the program:";
    int endKey = 0;
    std::cin >> endKey;
    return returncode;
}

