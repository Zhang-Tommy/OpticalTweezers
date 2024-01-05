#pragma once


char* read_file(std::string filePath);
void initialize_holo_engine();
int send_message(char* message);
int update_uniform(int uniform_var, const std::vector<float>& values, int num_values);
