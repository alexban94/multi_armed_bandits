#include <iostream>
#include <random>

class BanditProblem {
    private:
        int iterations, t, epsilon, k;
        std::vector<double> q, q_a;
        std::vector<int> n;

        std::default_random_engine generator;

    public:
    
    // Constructor.
    BanditProblem(int k, int epsilon, int iterations){
        this->k = k;
        this->iterations = iterations;
        this->epsilon = epsilon;

        // Initialize n and q for k bandits/actions.
        n.assign(k, 0);
        q.assign(k, 0);  

        /* For this problem, each action a from 1 to k has an expected reward q*(a) =  E[R_t | A_t = a].
        R_t is the reward at timestep t, and A_t is the action taken at that timestep. To emulate a problem,
        q*(a) can be sampled from the Normal distrubtion. */
        
        std::normal_distribution<double> distribution(0, 1);
        q_a.assign(k, 0);
        for(int i=0; i<k; i++){
            q_a[i] = distribution(generator);
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

    void run(){
        // Run the problem for specifed number of iterations.
        int a; // the action chosen at timestep t.
        double r; // the reward for choosing action a at timestep t.

        // Uniform distribution for chance of exploration.
        std::uniform_real_distribution<double> uniform_explore(0,1);
        // Uniform distribution for selecting a random action.
        std::uniform_int_distribution<int> uniform_int(0, k);

        for(int t=0; t<iterations; t++){
            // If probability < 1 - epsilon, use argmax Q(a) to select best action.
            if(uniform_explore(generator) < 1 - epsilon){
                
            } else {
                // Select a random action.
                a = uniform_int(generator);
            }
            // Obtain reward for action a at timestep t.
            r = bandit(a);

            // Update "step-size" parameter (1/n) for sample-average method of estimating action-values.
            n[a] += 1;
            q[a] = q[a] + ((1/n[a]) * (r - q[a]));
        }
    }

    double bandit(int action){
        // Sample from a normal distribution with mean = q*(a) for the chosen action a, to obtain reward for this step.
        std::normal_distribution<double> distribution(q_a[action], 1);
        return (double) distribution(generator);
    }

};

int main(){

    // Number of experiments to perform.
    const int n_experiments = 1000;

    // Run a single bandit problem.
    BanditProblem test_bandit = BanditProblem(10, 0.1, 1000);
    test_bandit.test();

    return 0;
}