---
layout: post
title: Unit Testing Python on GitHub
tag: 
- Python
- Software Development
- Unit Testing 
- DevOps
---

GitHub Actions is GitHub's platform for automating software workflows for continuous integration (CI) and continuous deployment (CD). We can use GitHub actions together with tools such as Pytest and Ruff for automated testing as part of a continuous integration process. In this post, I will explain how to use Pytest and Ruff for unit testing in Python, and how to automate this with GitHub actions as a branch protection rule in order to protect repositories from the effects of regression by enforcing test completion prior to merging of pull requests. 

# Unit Testing with Pytest

Pytest is a testing framework for Python which facilitates creating unit tests. 

Pytest may be installed using the package installer for Python with the following command:

```
pip install pytest
```

We can implement a Python unit test by creating a Python source file with a filename beginning with the characters "test_". If Pytest is run in the directory containing files with this naming pattern it will execute the functions defined in the source file in which we can implement unit and integration tests. Successful executing of the functions within these files results in successful completion of the test.

Take the following Python source file for example:

{% highlight python %}
#!/usr/bin/env python
"""test_PowerNetwork.py: unit tests for PowerNetwork class."""

# Import built-in modules
import math
import sys

# Import custom modules
sys.path.append('../')
from estimator import PowerNetwork

def test_StateEstimateLine():

    # Create power network object
    net = PowerNetwork()

    # Create buses
    busPoc = net.addBus("PoC Bus", 330, True)
    busHv1 = net.addBus("HV Bus 1", 330)
    
    # Create line
    Sbase = 100
    Zbase = (330000)**2/(Sbase*1000000)
    net.addLine("Line 1", busPoc, busHv1, 0, 0, 1/(-2*math.pi*1*50/Zbase), 0)

    # Add measurements
    net.addMeasurement("PoC", "V", 1, busPoc)
    net.addMeasurement("PoC", "P", 0.9, busPoc, busHv1)
    net.addMeasurement("PoC", "Q", 0.3, busPoc, busHv1)
    
    # Estimate states
    states = net.stateEstimate()

    # Correct type
    assert(type(states) is dict)
    # Correct length
    assert(len(states) == 2)

    # Check for expected states
    Expected0 = {'name': 'PoC Bus', 'vnominal': 330, 'magnitude': 1.0, 'phase': 0.0}
    assert(states[0] == pytest.approx(Expected0, rel=1e-6, abs=1e-12))
    Expected1 = {'name': 'HV Bus 1', 'vnominal': 330, 'magnitude': 0.9496367096114429, 'phase': -0.2769313269844059}
    assert(states[1] == pytest.approx(Expected1, rel=1e-6, abs=1e-12))

    return

{% endhighlight %}

This is one of the unit tests for the *PowerNetwork* class [in this repository](https://github.com/bott-j/pydbfilter/tree/main). The file contains one function, *test_StateEstimateLine()*, which tests state estimation functionality for a simple network containing one line. 

The function creates a network using the *PowerNetwork* class imported from the *estimation* module, and then estimates the bus voltages using the *stateEstimate()* method. 

The following lines check the type and dimension of the return value from this method.

{% highlight python %}
    # Correct type
    assert(type(states) is dict)
    # Correct length
    assert(len(states) == 2)
{% endhighlight %}

In this case the assert statement is used to test the required condition that the result has two elements and is a dictionary type. If the type or shape of the return value is not as expected an *AssertionError* exception will be raised and the unit test will fail.

We also use list comparison to verify the values in the return value, as shown below. In this case the expected value is first passed to the *approx()* function from the *pytest* package. This allows some tolerance for error in the comparison with the expected value, as we could reasonably expect variations in the result with a numerical method.

{% highlight python %}

    # Check for expected states
    Expected0 = {'name': 'PoC Bus', 'vnominal': 330, 'magnitude': 1.0, 'phase': 0.0}
    assert(states[0] == pytest.approx(Expected0, rel=1e-6, abs=1e-12))
    Expected1 = {'name': 'HV Bus 1', 'vnominal': 330, 'magnitude': 0.9496367096114429, 'phase': -0.2769313269844059}
    assert(states[1] == pytest.approx(Expected1, rel=1e-6, abs=1e-12))

{% endhighlight %}

If we run Pytest from the directory containing this source file, we get the following output:

```
>pytest
================================================= test session starts =================================================
platform win32 -- Python 3.8.5, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\repos\python-state-estimation\python-state-estimation\test
plugins: cov-5.0.0
collected 1 item

test_PowerNetwork.py .                                                                                           [100%]

================================================== 1 passed in 0.59s ==================================================
```

From the output we can see that one test passed. Under Windows we can check the exit code of the process using the *errorLevel* environment variable (we could also display the exit code under the Bash shell in Linux using "echo $?").

```
>echo %errorLevel%
0
```

In this case we can see that the exit code of the process was 0, indicating no errors.

# Linting with Ruff

Ruff is a linting tool for python which is used to check code quality by identifying issues in code style or potential coding errors.

Install ruff with the package installer for Python:

```
pip install pytest
```

Ruff works similarly to Pytest, to use Ruff we need to specify which files should be checked. For example, we can run Ruff to lint a single file by specifying the check command and a filename:

```
>ruff check test_PowerNetwork.py
test_PowerNetwork.py:47:12: E721 Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks
   |
46 |     # Correct type
47 |     assert(type(states) == dict)
   |            ^^^^^^^^^^^^^^^^^^^^ E721
48 |     # Correct length
49 |     assert(len(states) == 2)
   |

Found 1 error.
```

In this case Ruff identified a codestyle error in a type comparison and suggested a resolution. 

Repeating Ruff after resolving the type-comparison shows all checks passed:

```
>ruff check stateEstimator.py
All checks passed!
```

We could also pass "." to Ruff in place of a filename to check all files in the current directory.

# Configuring a Workflow with Github Actions

GitHub workflows are configured by placing a YAML file under the ".github/workflows/" directory tree in a repository. YAML (YAML Ain't Markup Language) is a human-readable data serialization language [1] similar to JSON or XML which is often used for configuration. The YAML configuration will identify what jobs need to be run in a workflow, how to setup the environment, and the conditions for the workflow to be executed. When a workflow is triggered it is executed in a runner under a virtual machine, which are provided free on GitHub for public repositories. 

For this example, we will create a unit testing workflow which will run when a pull request is created or updated for the main branch, and name the file "test-workflow.yaml".

The first key we will define is "name", which sets the name of the workflow, which for this example will be "Test Workflow".

```
name: Test Workflow
```

Next we can set the condition to run the workflow under the 'on' keyword:

```
on:
  pull_request:
    branches:
      - main
```

We will create the "jobs" key and define "test-job" as a job which wil run under the workflow. The environment is specified under the "runs-on" key, for which we have specified the latest version supported of Ubuntu linux. The "run:working-directory" key specifies the working directory where the Pytest files will be contained, which is "./test/" in this repository.

```
jobs:
  test-job:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ./test/
```

For the "strategy" key we set "matrix", which causes the workflow to be run under the environments specified. In this case only one environment is specified which is Python 3.11. 

```
    strategy:
      matrix:
        python-version: [3.11]
```

The steps key specifies each of the steps to be run in the job. The first item in the list, "uses", is a repository checkout action. The second item named "Setup Python" sets up the Python version to be used in the testing environment using the setup python action.

```
  steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{matrix.python-version}}
```

Next, the dependencies are installed in Python. The "run" key specifies the command to run within the environment. The two commands upgrade PIP and install the package versions specified in the "requirements.txt" file which we will talk more about later.

```
 - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
```

Finally, the testing and linting steps to be carried out are specified. 

```
    - name: Test python
        run: |
          pytest -vv
          pytest --cov --cov-fail-under=90
      - name: Lint python
        run: ruff check --output-format=github .
```

Next we will want to create the requirements file mentioned previously. This is just a text file which specifies the Python package versions which need to be installed in our test environment. In our case this is a file named "requirements.txt" which is placed under the "test/" directory created in the repository.

```
numpy==2.1.3
pandas==2.2.3
pytest==8.3.4
pytest-cov==6.0.0
ruff==0.8.2
```

Once the workflow YAML configuration has been copied to the main branch the workflow we defined earlier should be executed whenever a pull request is created or updated. The history of these workflows can be viewed under the Actions tab of the repository on GitHub.

For example below we can see two workflow runs under the Actions tab. Each workflow run is named according to the pull request description. The coloured icon adjacesn each workflow run identifies the success or failure of the run.

![GitHub actions tab showing a failed workflow run in red and a successful workflow run in green.](/assets/posts/unit-testing-python-on-github/unit-testing-8.png?raw=true)

# Setup Branch Protection Rules

While the workflow defined previously runs automatically, we do nothing with the testing and linting results at this stage. The next step is to setup branch protection rules to ensure the workflow completes successfully before allowing a pull request to be merged.

Branch protection rules can be setup by through the Settings tab in the repository. Selecting Rules -> Rulesets under the Code and Automation group opens the Rulesets configuration page. From there we can create a new branch ruleset from the drop-down menu as shown below.

![Creating a new Ruleset on the rulesets configuration page under repository settings.](/assets/posts/unit-testing-python-on-github/unit-testing-9.png?raw=true)

We can name this rule-set "test-ruleset" in this example and set the enforcement status to Active to enable the rule.

![Setting ruleset name and enforcement status.](/assets/posts/unit-testing-python-on-github/unit-testing-10.png?raw=true)

We will also need to specify the target branch under "Target Branches" using the "Add Target" drop-down menu. This will apply the ruleset only to the main branch, as we want to protect merges from development branches into main. This can be done using the "Include by pattern" menu item.

![Setting a text pattern to determine the target branch.](/assets/posts/unit-testing-python-on-github/unit-testing-2.png?raw=true)

Under the Rules heading, Enforce pull requests by selecting "Require a pull request before merging". The number of required appovals can also be set here, however no one else is working on the repository then required approvals could be set to zero.

![Pull request and appoval requirements.](/assets/posts/unit-testing-python-on-github/unit-testing-3.png?raw=true)

Then we can select "Require Status Checks to Pass", which will allow us to specify the jobs which need to be completed before allowing a pull request to be merged. It is a good idea to select "Require branches to be up to date before merging" also, to require changes from the target branch to be incorporated into the development branch.

![Status check and branch merging settings.](/assets/posts/unit-testing-python-on-github/unit-testing-4.png?raw=true)

The status checks are selected using the "Add Checks" button and typing in the name of our job as a search pattern. In this case we select the job named "test-job". The Workflow definition YAML needs to have been merged into the main branch for the job to be visible here.

![Setting GitHub Actions jobs as required checks on the branch.](/assets/posts/unit-testing-python-on-github/unit-testing-11.png?raw=true)

After this the ruleset can be created using the "Create" button at the bottom of the settings page.

# Branch Protection in Action

On creating a pull request, we can see that checks have failed:

![Failed checks visible on the Pull Request page.](/assets/posts/unit-testing-python-on-github/unit-testing-12.png?raw=true)

The details link provides information on the job including output logs. In this case we can see that the "test_StateEstimationLine()" function failed within the "test_PowerNetwork.py" file, due to an unexpected result in the comparison.

![Details of job run showing output logs from Pytest.](/assets/posts/unit-testing-python-on-github/unit-testing-13.png?raw=true)

If we now correct the error causing the unit test to fail, then commit and push the updated file, we can see that all tests have passed. 

![Status indicating passed checks on the Pull Request page.](/assets/posts/unit-testing-python-on-github/unit-testing-14.png?raw=true)

At this point the pull request can be merged, having successfully resolved the failed unit test.
