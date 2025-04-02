#pragma once
#include <iostream>
#include <string>
#include <regex>
#include <map>
#include <fstream>
#include <sstream>
#include <istream>

class Parser
{
private:
    std::string _filePath;
    std::map<std::string,std::string> _container;
    void parser(std::string& line);
public:
    Parser(std::string& filePath);
    Parser(std::string&& filePath);
    ~Parser();
    void loader();
    std::map<std::string,std::string> getting_data();
    void print();
};
Parser::Parser(std::string& filePath) : _filePath(filePath){}
Parser::Parser(std::string&& filePath) : _filePath(filePath){}
Parser::~Parser(){}
void Parser::loader()
{
    std::string line;
    std::ifstream file(_filePath);
    if (!file)
    {
        std::cout << "file is not opened yet";
    }
    else {
        while(std::getline(file,line))
        {
            parser(line);
        }
    }
    file.close();
}
std::map<std::string,std::string> Parser::getting_data()
{
    return _container;
}
void Parser::print()
{
    for (auto& it : _container)
    {
        std::cout << it.first << std::endl;
    }
}
void Parser::parser(std::string& line)
{
    std::regex reg(R"(^\s*([\w_]+)\s*=\s*(.+)\s*$)");
    std::smatch match;
    if (std::regex_search(line,match,reg))
    {
        _container[match[1]] = match[2];
    }
}
bool stringToDouble(const std::string& str, double& num) 
{
    std::stringstream ss(str);
    ss >> num;
    
    // Check if conversion was successful
    return !ss.fail() && ss.eof();
}


