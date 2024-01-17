#include <cstring>
#include <iostream>
#include <memory>


#include "ExplicitStateDfaMona.h"
#include "InputOutputPartition.h"
#include "Preprocessing.h"
#include <lydia/parser/ltlf/driver.hpp>
#include <CLI/CLI.hpp>
#include <istream>


int main(int argc, char ** argv) {

    std::chrono::high_resolution_clock::time_point start_time;

    CLI::App app {
            "LTLfToDFA: A tool for obtaining the minimized DFA of a given LTLf formula"
    };

    std::string formula_file;

    app.add_option("-f,--spec-file", formula_file, "Specification file")->
                    required() -> check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    std::ifstream in(formula_file);
    std::string f;
    std::getline(in, f);


    // Parsing the formula
    std::shared_ptr<whitemech::lydia::parsers::ltlf::LTLfDriver> driver;
    driver = std::make_shared<whitemech::lydia::parsers::ltlf::LTLfDriver>();
    std::stringstream formula_stream(f);
    driver->parse(formula_stream);
    whitemech::lydia::ltlf_ptr parsed_formula = driver->get_result();
    // Apply no-empty semantics
    auto context = driver->context;
    auto not_end = context->makeLtlfNotEnd();
    parsed_formula = context->makeLtlfAnd({parsed_formula, not_end});



    Syft::ExplicitStateDfaMona explicit_dfa_mona = Syft::ExplicitStateDfaMona::dfa_of_formula(*parsed_formula);
    explicit_dfa_mona.dfa_print();

    std::chrono::high_resolution_clock::time_point stop_time;
    std::chrono::milliseconds total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop_time - start_time);
    std::cout << "Total time: "
              << total_time.count() << " ms" << std::endl;


//    Syft::ExplicitStateDfa explicit_dfa = Syft::ExplicitStateDfa::from_dfa_mona(var_mgr, explicit_dfa_mona);


  return 0;
}

