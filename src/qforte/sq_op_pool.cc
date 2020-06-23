#include "helpers.h"
#include "quantum_gate.h"
#include "quantum_circuit.h"
#include "quantum_operator.h"
#include "sq_operator.h"
#include "sq_op_pool.h"

#include <stdexcept>
#include <algorithm>

void SQOpPool::add_term(std::complex<double> coeff, const SQOperator& sq_op ){
    terms_.push_back(std::make_pair(coeff, sq_op));
}

void SQOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "Number of new coeficients for quantum operator must equal." );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

const std::vector<std::pair< std::complex<double>, SQOperator>>& SQOpPool::terms() const{
    return terms_;
}

void SQOpPool::set_orb_spaces(const std::vector<int>& ref){
    int norb = ref.size();
    if (norb%2 == 0){
        norb = static_cast<int>(norb/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of spin orbitals.");
    }

    nocc_ = 0;
    for (const auto& occupancy : ref){
        nocc_ += occupancy;
    }

    if (nocc_%2 == 0){
        nocc_ = static_cast<int>(nocc_/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of occupied spin orbitals.");
    }

    nvir_ = static_cast<int>(norb - nocc_);
}

std::vector<QuantumOperator> SQOpPool::get_quantum_operators(){
    std::vector<QuantumOperator> A;
    for (auto& term : terms_) {
        QuantumOperator a = term.second.jw_transform();
        a.mult_coeffs(term.first);
        A.push_back(a);
    }
    return A;
}

QuantumOperator SQOpPool::get_quantum_operator(){
    QuantumOperator A;
    for (auto& term : terms_) {
        QuantumOperator a = term.second.jw_transform();
        a.mult_coeffs(term.first);
        A.add_op(a);
    }
    // TODO: analyze ordering here, eleimenating simplify will place comuting
    // terms closer together but may introduce redundancy.
    A.simplify();
    return A;
}

void SQOpPool::fill_pool(std::string pool_type){
    if(pool_type=="sa_SD"){
        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for (size_t a=0; a<nvir_; a++){
                size_t aa = 2*nocc_ + 2*a;
                size_t ab = 2*nocc_ + 2*a+1;

                SQOperator temp1;
                temp1.add_term(+1.0/std::sqrt(2), {aa, ia});
                temp1.add_term(+1.0/std::sqrt(2), {ab, ib});

                temp1.add_term(-1.0/std::sqrt(2), {ia, aa});
                temp1.add_term(-1.0/std::sqrt(2), {ib, ab});

                temp1.simplify();

                std::complex<double> temp1_norm(0.0, 0.0);
                for (const auto& term : temp1.terms()){
                    temp1_norm += std::norm(term.first);
                }
                temp1.mult_coeffs(1.0/std::sqrt(temp1_norm));
                add_term(1.0, temp1);
            }
        }

        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for(size_t j=i; j<nocc_; j++){
                size_t ja = 2*j;
                size_t jb = 2*j+1;

                for(size_t a=0; a<nvir_; a++){
                    size_t aa = 2*nocc_ + 2*a;
                    size_t ab = 2*nocc_ + 2*a+1;

                    for(size_t b=a; b<nvir_; b++){
                        size_t ba = 2*nocc_ + 2*b;
                        size_t bb = 2*nocc_ + 2*b+1;

                        SQOperator temp2a;
                        if((aa != ba) && (ia != ja)){
                            temp2a.add_term(2.0/std::sqrt(12), {aa,ba,ia,ja});
                        }
                        if((ab != bb ) && (ib != jb)){
                            temp2a.add_term(2.0/std::sqrt(12), {ab,bb,ib,jb});
                        }
                        if((aa != bb) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb,ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba,ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb,ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba,ia,jb});
                        }

                        // hermetian conjugate
                        if((ja != ia) && (ba != aa)){
                            temp2a.add_term(-2.0/std::sqrt(12), {ja,ia,ba,aa});
                        }
                        if((jb != ib ) && (bb != ab)){
                            temp2a.add_term(-2.0/std::sqrt(12), {jb,ib,bb,ab});
                        }
                        if((jb != ia) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia,bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib,ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib,bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia,ba,ab});
                        }

                        SQOperator temp2b;
                        if((aa != bb) && (ia != jb)){
                            temp2b.add_term(0.5, {aa,bb,ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2b.add_term(0.5, {ab,ba,ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2b.add_term(-0.5, {aa,bb,ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2b.add_term(-0.5, {ab,ba,ia,jb});
                        }

                        // hermetian conjugate
                        if((jb != ia) && (bb != aa)){
                            temp2b.add_term(-0.5, {jb,ia,bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2b.add_term(-0.5, {ja,ib,ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2b.add_term(0.5, {ja,ib,bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2b.add_term(0.5, {jb,ia,ba,ab});
                        }

                        temp2a.simplify();
                        temp2b.simplify();

                        std::complex<double> temp2a_norm(0.0, 0.0);
                        std::complex<double> temp2b_norm(0.0, 0.0);
                        for (const auto& term : temp2a.terms()){
                            temp2a_norm += std::norm(term.first);
                        }
                        for (const auto& term : temp2b.terms()){
                            temp2b_norm += std::norm(term.first);
                        }
                        temp2a.mult_coeffs(1.0/std::sqrt(temp2a_norm));
                        temp2b.mult_coeffs(1.0/std::sqrt(temp2b_norm));

                        if(temp2a.terms().size() > 0){
                            add_term(1.0, temp2a);
                        }
                        if(temp2b.terms().size() > 0){
                            add_term(1.0, temp2b);
                        }
                    }
                }
            }
        }
    } else if(pool_type == "test"){
        SQOperator A;
        A.add_term(+2.0, {1,2,4,3});
        A.add_term(-2.0, {2,4});
        A.add_term(-2.0, {3,4,2,1});
        A.add_term(+2.0, {4,2});
        add_term(-0.25, A);
        add_term(+0.75, A);
    } else {
        throw std::invalid_argument( "Invalid pool_type specified." );
    }
}

std::string SQOpPool::str() const{
    std::vector<std::string> s;
    s.push_back("");
    int counter = 0;
    for (const auto& term : terms_) {
        s.push_back("----->");
        s.push_back(std::to_string(counter));
        s.push_back("<-----\n");
        s.push_back(to_string(term.first));
        s.push_back("[\n");
        for (const auto& sub_term : term.second.terms()) {
            int nbody = sub_term.second.size() / 2.0;
            s.push_back(to_string(sub_term.first));
            s.push_back("(");
            for (int k=0; k<nbody; k++ ) {
                s.push_back(std::to_string(sub_term.second[k]) + "^");
            }
            for (int k=nbody; k<2*nbody; k++ ) {
                s.push_back(std::to_string(sub_term.second[k]));
            }
            s.push_back(")\n");
        }
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}
