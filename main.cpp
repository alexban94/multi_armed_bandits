#include <iostream>
#include <random>
#include <algorithm> // max_element
#include <cmath> // abs
#include <vector>


struct BanditResults {
        double epsilon;
        std::vector<double> optimal_action;
        std::vector<double> reward_log;
    };

class BanditProblem {
    private:
        int iterations, t, k;
        double epsilon;
        std::vector<double> q, q_a;
        std::vector<int> n;

        std::default_random_engine generator;
        std::vector<std::normal_distribution<double>> reward_distributions;

    public:
    
    // Constructor.
    BanditProblem(int k, double epsilon, int iterations, unsigned int seed = 0){
        this->k = k;
        this->iterations = iterations;
        this->epsilon = epsilon;

        // Initialize n and q for k bandits/actions.
        n.assign(k, 0);
        q.assign(k, 0);  

        /* For this problem, each action a from 1 to k has an expected reward q*(a) =  E[R_t | A_t = a].
        R_t is the reward at timestep t, and A_t is the action taken at that timestep. To emulate a problem,
        q*(a) can be sampled from the Normal distrubtion. */
        generator.seed(seed);
        std::normal_distribution<double> distribution(0, 1);
        q_a.assign(k, 0);
        for(int i=0; i<k; i++){
            q_a[i] = distribution(generator);
            reward_distributions.push_back(std::normal_distribution<double>{q_a[i], 1});
        }        

    }

    void test(){
        /*for(t=0; t<iterations; t++){
            std::cout << "Iteration: " << t << "\n" << std::endl;
        }*/
        for(int i=0; i<k; i++){
            std::cout << "a, n and q: " << i << " " << n[i] << " " << q[i] << std::endl;
        }
        for(int i=0; i<k; i++){
            std::cout << "q*(a) values: a= " << i << " q*(a)=  " << q_a[i] << std::endl;
        } 
        // Check vector size.
        std::cout << "Size of n: " << n.size() << " Size of q: " << q.size() << std::endl;

        // Test bandit.
        for(t=0; t<iterations; t++){
            std::cout << "Iteration: " << t << ", Action: 3, R_t = "<< bandit(0) << std::endl;
        }
    }

    BanditResults run(){
        // Run the problem for specifed number of iterations.
        int a = 0; // the action chosen at timestep t.
        int best_a = 0; // the best possible action chosen at timestep t.
        double r[k] = {0}; // the rewards for choosing action a at timestep t.

        // Statistics for evaluation.
        std::vector<double> reward_log(iterations, 0);
        std::vector<double>  optimal_action(iterations, 0);

        bool optimal_a[iterations] = {false}; // True if the optimal action was taken at timestep t.
        double reward_at_t[iterations] = {0};

        // Uniform distribution for chance of exploration.
        std::uniform_real_distribution<double> uniform_explore(0,1);
        // Uniform distribution for selecting a random action.
        std::uniform_int_distribution<int> uniform_int(0, k-1); // a <= sample <= b, so need to sample up to k-1.

        int count_random = 0;
        int count_chosen = 0;

        // Testing intial values:
        /* std::cout<<"Initial values:"<<std::endl;
        for(int i=0; i<k;i++){
            std::cout<<"r["<<i<<"] = "<<r[i]<<", q["<<i<<"] = "<<q[i] <<" ;"<<std::endl;
        }
        for(int i=0; i<k;i++){
            std::cout<<"q_a["<<i<<"] = "<<q_a[i] <<" ;"<<std::endl;
        }*/

        for(int t=0; t<iterations; t++){
            //std::cout<<"Timestep: "<<t<<std::endl;
            // If probability < 1 - epsilon, use argmax Q(a) to select best action.
            if(uniform_explore(generator) < 1.0 - epsilon && t > 0){
                //std::cout<<"Selected"<<std::endl;
                a = std::distance(q.begin(), std::max_element(q.begin(), q.end()));
                count_chosen += 1;

            } else {
                // Select a random action.
                //std::cout<<"Random"<<std::endl;
                a = uniform_int(generator);
                count_random += 1;
            }

            //for(int i=0; i < q.size(); i++)
                //std::cout<<"q["<<i<<"] = "<<q[i] <<" ;";
            //std::cout<<std::endl;

            // Obtain rewards for all actions at timestep t - for the purpose of evaluating whether the best action was taken.
            for(int i = 0; i < k; i++){
                r[i] = bandit(i);
               // std::cout<<"r["<<i<<"] = "<<r[i] <<" ;";
            }
            //std::cout<<std::endl;

            // Find best action in r - the argmax.
            best_a = std::distance(r, std::max_element(r, r + k));
            //DEBUG:
            //std::cout<<"Action taken: "<<a<<" Reward: "<<r[a];
            //std::cout<<" Optimal action: "<<best_a<<" Reward: "<<(*std::max_element(r, r + k))<<std::endl;

            // If best_a is equal to a, then the best action was selected.
            if(best_a == a){
                optimal_a[t] = true;
            }
            // TODO: fix this.
            // Update avg_optimal_action using optimal a. This stores the current % accuracy the best action was taken at each t.
            if(t > 0){
                optimal_action[t] = ((double)std::accumulate(optimal_a, optimal_a + t, 0)) * 100/t; //  Sum from first element up to current element at time t. Divide by t for accuracy.
                //std::cout<<"Test accumulate"<<std::endl;
                //std::cout<<((double)std::accumulate(optimal_a, optimal_a + t, 0))/t<<std::endl;
            } else {
                // Avoid division by 0 on first timestep. 
                optimal_action[t] = ((double) std::accumulate(optimal_a, optimal_a + t, 0)) * 100;
                
            }
            // Add r[a] to avg_reward for this timestep t.
            reward_log[t] = r[a];
            

            
            // Only the reward r[a] of the chosen action a is used in updating the corresponding N(a) and Q(a) on this timestep.
            // Update "step-size" parameter (1/n) for sample-average method of estimating action-values.
            //std::cout<<"Before: N[a] = "<<n[a]<<" q[a] = "<<q[a];
            n[a] += 1;
            //double r_minus_q, rq_divide_n, rqn_plus_q, one_over_n = 0;
            //r_minus_q = r[a] - q[a];
            //one_over_n = 1.0/n[a];
            //rq_divide_n = r_minus_q * one_over_n;
            //rqn_plus_q = q[a] + rq_divide_n;
            //std::cout<<"r[a] - q[a] = "<<r_minus_q<<std::endl;
            //std::cout<<"1/n[a] = "<<one_over_n<<std::endl;
            //std::cout<<"1/n[a] * (r[a] - q[a]) = "<<rq_divide_n<<std::endl;
            //std::cout<<"q[a] +     1/n[a] * (r[a] - q[a]) = "<<rqn_plus_q<<std::endl;

            q[a] = q[a] + ((1.0/n[a]) * (r[a] - q[a]));
            //std::cout<<" After: N[a] = "<<n[a]<<" q[a] = "<<q[a]<<std::endl<<std::endl;
        }
        
        std::cout<<"Total correct actions taken: "<<std::accumulate(optimal_a, optimal_a + iterations, 0)<<std::endl;
        std::cout<<"Epsilon: "<<epsilon<<" Iterations: "<<iterations<<std::endl;
        std::cout<<"Total actions selected: "<<count_chosen<<std::endl;
        std::cout<<"Total random actions taken: "<<count_random<<std::endl;
        
        BanditResults result;
        result.optimal_action = optimal_action;
        result.reward_log = reward_log;
        result.epsilon = epsilon;
        return result;
        
    }

    double bandit(int action){
        // Sample from a normal distribution with mean = q*(a) for the chosen action a, to obtain reward for this step.
        //std::normal_distribution<double> distribution(q_a[action], 1);
        //return (double) distribution(generator);
        return reward_distributions[action](generator);
    }

};

void print_stats(std::vector<double> &avg_reward, std::vector<double> &avg_optimal_action){
    std::cout<<"Average reward over 1000 timesteps:"<< std::endl;
    for(int i=0; i < avg_reward.size(); i++)
        std::cout <<i<<": "<<avg_reward.at(i) <<std::endl;
    std::cout<<"Average optimal action accuracy over 1000 timesteps:"<< std::endl;
    for(int i=0; i < avg_optimal_action.size(); i++)
        std::cout <<i<<": "<<avg_optimal_action.at(i) <<"\%"<<std::endl;
}

// Prints progress of averaged results every i iterations.
void print_tabular(std::array<BanditResults> result_array, int interval=100, int timesteps, int n_experiments){
   std::cout<<"Average reward over "<<n_experiments<<"runs using "<<timesteps<<" timesteps:"<<std::endl;
   std::cout<<"----------------------------------------------"<<std::endl;
   std::cout<<"timestep\t\t";
   for(int i=0; i<result_array.size(); i++){
      // Print columns using epsilon values.
      std::cout<<"\t"<<result_array[i].epsilon;
   }
   
   for(int i=0; i <= timesteps-1; i+interval){
      std::cout<<"t=
   }
}

// Runs all n_experiments and returns the averaged results.
BanditResults run_experiment_config(int k, double epsilon, int timesteps, int n_experiments){
    // Vectors to tally average results.
    std::vector<double> avg_reward(timesteps,0), avg_optimal_action(timesteps,0);
    
    BanditResults result;
    // Run bandit trials.
    for(int n=0; n<n_experiments; n++){
        std::cout<<"Trial "<<n<<std::endl;
        result = BanditProblem(10, 0.1, timesteps, n).run();
        // Add current results to the total.
        for(int i=0; i<timesteps; i++){
            avg_optimal_action[i] += result.optimal_action[i];
            avg_reward[i] += result.reward_log[i];
        }
    }
    // Average results by dividing by n_experiments.
    // First two parameters give the range of elements to transform.
    // Third parameter is the intial iterator to where to store values after transform.
    // Fourth is the lambda function: takes n_experiments as a "captured" variable from the enclosing scope;
    // takes a double  reference as a parameter (each vector element). Followed by the function, returning the division.
    transform(avg_optimal_action.begin(), avg_optimal_action.end(), avg_optimal_action.begin(), [n_experiments](double &c){ return c/n_experiments; });
    transform(avg_reward.begin(), avg_reward.end(), avg_reward.begin(), [n_experiments](double &c){ return c/n_experiments; });
        
    BanditResults avg_result;
    avg_result.optimal_action = avg_optimal_action;
    avg_result.reward_log = avg_reward;
    avg_result.epsilon = epsilon;
    return avg_result;
  
}

int main(){

    // Number of experiments to perform.
    const int n_experiments = 2000;
    const int timesteps = 1000;


    // Run a single bandit problem.
    BanditProblem test_bandit = BanditProblem(10, 0.1, timesteps);
    BanditResults test_result = test_bandit.run();
    // Print out statistics.
    //print_stats(test_result.reward_log, test_result.optimal_action);

    // Peform trial experiments.
    int k = 10;
    BanditResults greedy = run_experiment_config(k, 0, timesteps, n_experiments);
    BanditResults small_epsilon = run_experiment_config(k, 0.01, timesteps, n_experiments);
    BanditResults large_epsilon = run_experiment_config(k, 0.1, timesteps, n_experiments);
        
    std::array<BanditResults> result_array = {greedy, small_epsilon, large_epsilon};    
    print_tabular(result_array);
    //print_stats(avg_reward, avg_optimal_action);
    
    return 0; 
}


 /*
    int v[10] = {1, 2, 3, 4 ,7, 5, 9, 6, 4, 2};
    std::vector<double> vec = {3.2, 4.6, 5.6, 1.2, 4.4};

    int arg_max = std::distance(v, std::max_element(v, v + 10));
    int vec_arg_max = std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));

    std::cout << "Max array: " << arg_max <<std::endl;
    std::cout << "ArgMax vector: " << vec_arg_max <<std::endl;
    */
