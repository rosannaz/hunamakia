
# coding: utf-8

# In[1]: Setting up

get_ipython().magic('pylab inline')

from oauth2client.client import GoogleCredentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import uuid
import time

import func_bigquery as bq
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import numpy as np

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import Levenshtein
import math

# Query from BigQuery--------------------------------------------------------

def get_credentials():
    project_id = 'leafy-tenure-393'
    credentials = GoogleCredentials.get_application_default()
    bigquery = build('bigquery', 'v2', credentials=credentials)
    return(bigquery,project_id)

def execute_async_query(query,dataset_id,table_id,createDisposition = 'CREATE_IF_NEEDED', writeDisposition = 'WRITE_TRUNCATE'):
    bigquery,project_id = get_credentials()  
    query_response = bq.async_query(bigquery, project_id, query, dataset_id, table_id, createDisposition, writeDisposition)
    pprint.pprint(query_response)
    print('\n')
    bq.poll_job(bigquery, query_response)
    print('\n')
    
def execute_sync_query(query):
    bigquery,project_id = get_credentials()  
    data = bq.sync_query(bigquery,project_id,query)
    return(data)


# In[90]:

# Select data from BigQuery--------------------------------------------------


def convertData (query):
    data = execute_sync_query(query)
    field1 = [] #new list
    field2 = [] #list2

    #create two separate lists by elements of the data list
    for i in range(len(data)):
        field1.append(data[i][0]) #nTitle
        field2.append(data[i][1]) #cTitle
    
    return ([field1, field2])


#Remove punctuations and lowercase------------------------------------
def cleanTitle( title ):
    for i in range (len (title)):
        letters_only = re.sub("[,-./()]", " ", title[i])
        words = letters_only.lower().split()
        title[i] = ( " ".join( words ))
        
    return title


#Vectorize words to array-----------------------------------------------------

def vectorize(x):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    
    vecX = vectorizer.fit_transform(x)
    #convert to array
    vecX = vecX.toarray()

    print('Finished vectorizer...', vecX)
    return vecX


# Construct MultinomialNB Model------------------------------------------------
def MultinomialNBModel (x, y):
    
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    
    print ('Running fit model')
    # Train the model using the training sets 
    model.fit(x, y)

    print ('Finished running')
    return model

#Predict possibilitity for each title-----------------------------------------


# Map possibility to term
def probDict(y, predictProb):

    vocab = {}
    y = np.unique(y) #unique y (next titles)

    #create dictionary of y:probability
    for i in range(len(predictProb)):

    #filter out low probabilities
        if predictProb[i] > 0.001:
            vocab[y[i]] = predictProb[i]

    return vocab
    #print (predicted, vocab)

#Combined def -----------------------------------------------------------------------
def genPredictModel(query):
    print ('Start generating')
    #prepare two lists of data x and y; in reverse here since current Title/Company is in field 1
    data = convertData(query)
    x = data[1]
    y = data[0] 

    print('Finished loading..')

    #clean list
    x = cleanTitle(x)
    y = cleanTitle(y)
    print('Finished cleaning', x[:50])
    #vectorize x
    x = vectorize(x)
    
    print('Finished vectorize', x[:50])
    #create prediction model
    model = MultinomialNBModel (x, y)

    #return model and y
#    return [model, np.unique(y)]
    return [model, np.unique(y), x]

def predictDictProb(model, inputVal, uy): #uy= unique y
    
    predicted = model.predict_proba(inputVal)
    predicted = predicted[0]
    probdict = probDict(uy, predicted)
    #return dictionary of y:probability
    return probdict



#criteria for jobs
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#create dictionary of y parameters, name of the next company/title to the index------------------------------------------------
'''def createDict (uy):
    Dict = {}
    for i in range(len(uy)):
        Dict[uy[i]] = i
    return Dict'''

#calculate similarity between jobs and potential jobs (next company/title)--------------------------------------------------
#returns list of probability (each row is a potential job's match with jobs in database)
def probSimScore (values, predictDictProb): #values = jobslist, data = potential jobs/uy
    probability = [] #combined probability
    data = predictDictProb.keys()
    for i in range(len(values)):
        iDict = {} #individual dict
        #for j in range(len(data)):
        for j in data:
            #filter out low probability, calculate ratio
            lev = Levenshtein.ratio(values[i], j)
            if lev > 0.0001:
                #iprobability.append(lev)
                iDict[j] = lev #dictionary of job to probability
        probability.append(iDict)
    
    return probability # job list of dictionary probability


#Calculate a job's probability combining the two factors 
#1)predicted probability of next title/company 2)similarity between the job descript and the predited job
def probScore (simScoreDict, predScoreDict):
    
    keys = predScoreDict.keys()
    probScoreList = []
    
    #create a dictionary of each potential job:probability of each
    #create a list containing scores of different jobs
    for i in range(len(simScoreDict)):
        probScoreDict = []
        keys2 = simScoreDict[i].keys()
        for j in keys:
            if j in keys2:
                probScoreDict.append(simScoreDict[i][j]*predScoreDict[j])
        probScoreList.append(max(probScoreDict))
        #returns maximum score of each job
    return probScoreList

#Calculate Descriptions/Skills Overlap:-----------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def calcDescriptOverlapScore(pD, jobD): #pD = string of personal description; jobD = list of job descriptions
    
    import Levenshtein
    stop_Words = text.ENGLISH_STOP_WORDS
    #Remove stop words first (the, etc.)
    def sanitize(user_input, stop_words):
        """Sanitize using standard list comprehension"""
        user_input = user_input.split()
        return ' '.join([w for w in user_input if w not in stop_words])
    
    pD = sanitize(pD.lower(), stop_Words)
    for i in range(len(jobD)):
        jobD[i] = sanitize(jobD[i].lower(), stop_Words)
    
    #calculate overlap score + create dictionary of jobD place:score

    descriptionList = []
    for i in range(len(jobD)):
        descriptionList.append(Levenshtein.ratio(jobD[i], pD))
        
    return descriptionList #list of each job's score

#Putting everything together:-----------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def sortedJobNum (descriptionList, probScoreListTitle, probScoreListCompany):
    
    #calcTotalScore: array of job# AND score
    scoreList = np.empty([1,2])
    for i in range(len(descriptionList)):
        scoreList = np.vstack((scoreList,np.array([i, (descriptionList[i]+math.sqrt(probScoreListTitle[i])+math.sqrt(probScoreListCompany[i]))/3])))
    scoreList = np.delete(scoreList, 0, 0)
    
    #Sort array by score
    scoreArray = scoreList[scoreList[:,1].argsort()]
    return scoreArray

def top10Jobs (scoreArray, jobT):
    top10JobList = []
    length = scoreArray.shape[0]
    for i in range(length):
        top10JobList.append(jobT[int(scoreArray[i][0])])
    return top10JobList
#updated

    #calcTotalScore: array of job# AND score
def jobRankSort(jobRank, jobD, jobTitle):
    #create array of [index, rank]
    scoreList = np.empty([1,2])
    for i in range(len(jobRank)):
        scoreList = np.vstack([scoreList,[i, jobRank[i]]])
    scoreList = np.delete(scoreList, 0, 0)
    
    #Sort array by score
    scoreArray = scoreList[scoreList[:,1].argsort()]

    #match index to score
    jobRankSorted = []
    for i in range(len(scoreArray)):
        jobRankSorted.append(jobTitle[int(scoreArray[i][0])])
    return jobRankSorted


# In[26]:

#Test cases:------------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

jobT = ["Information Technology Intern",
            "Technology Analyst - Intern",
            "PRINCIPAL MEMBER OF TECHNICAL STAFF",
            "IT Technology Consultant",
            "Real Estate Marketing Associate",
            "Marketing Assistant - Marketing Coordinator",
            "Professor - Electrical Engineering & Computer Science",
            "Software Engineer Amdocs",
            "Software Engineer",
            "Junior Trader"
            ]
jobT = cleanTitle(jobT)

jobC = ["John Deere",
            "ZS Associates",
            "AT&T Services",
            "SAP",
            "Berkshire Hathaway",
            "Atlantic Coast Events",
            "University of Cincinnati",
            "Amdocs Inc",
            "Harman",
            "Parchem"
        ]
jobC = cleanTitle(jobC)

#job description
jobD = ["Information Technology Intern As a Summer 2017 Information Technology Intern you will be provided with opportunities to partner in the planning and delivery of information technology to support business processes and business practices for strategic business units.  Information Technology Intern position duties:  Learn how to identify, analyze and apply information technology and business practices to support strategic business process/plans. Participate as required to design, develop, test and integrate technology. Participate in the implementation of information technology and business processes. Support, evaluate, and continuously improve information technology and business processes to maintain alignment with business plans. Perform activities accordingly to project plans and schedule. Has contact primarily focused around department and functional operations. Work Statement:  US Visa sponsorship is not available for this position.  What Skills You Need  Sophomore or Junior pursuing a Bachelors degree or Senior pursuing Masters in Information Technology, Computer Science, Computer Engineering, Management Information Systems or a related degree.",
        "Technology Analyst - Intern The Solution Delivery team implements technology based solutions in the areas of business intelligence, analytics, information management, reporting and mobility. In addition, this team configures ZS proprietary software used to streamline business processes for commercial groups at client organizations. This team takes overall ownership of the delivery of the technical solution leading the design and development phases and collaborating with other teams for business requirements and system testing.  Responsibilities:  Lead development phase during implementation of world-class technology solutions to solve business problems across one or more client engagements; Provide input to senior solutions delivery analyst and delivery lead to build comprehensive implementation project plans; Provide input to senior solutions delivery analyst based on past experience and understanding of requirements to design a flexible and scalable solution; Apply appropriate development methodologies (e.g.: agile, waterfall) and best practices (e.g.: mid-development client reviews, embedded QA procedures, unit testing) to ensure successful and timely completion; Collaborate with other team members (involved in the requirements gathering, testing, roll-out and operations phases) to ensure seamless transitions. Qualifications:  Enrollment in a bachelor's/master's degree with specialization in Computer or IT, BCA or other computer-related disciplines with a strong academic record; Experience working with front-end user reporting (e.g.: MSTR, Business Object, Cognos), back-end database management (e.g.: Oracle, Teradata) and/or ETL interfacing (e.g.: Informatica, SSIS) technologies. Additional skills:  Strong verbal and written communication skills with ability to articulate results and issues to internal and client teams Proven ability to work creatively and analytically in a problem-solving environment Ability to work within a virtual global team environment and contribute to the overall timely delivery of multiple projects Willingness to travel to other global offices as needed to work with client or other internal project teams ZS is a global consulting firm; English fluency is required, additional fluency in at least one European or Asian language is highly desired.",
        "PRINCIPAL MEMBER OF TECHNICAL STAFF needed by AT&T Services, Inc. in Bedminster, NJ to conduct leading-edge scientific and technological research in massive scale information mining solutions that support mission critical technologies for internal and commercial applications.",
        "IT Technology Consultant Part of the Business Insight & Technology (internal IT organization), the Enterprise Analytics team acts as an internal business partner by developing and deploying best in class Business Intelligence solutions to the entire SAP organization. This is done by leveraging SAPs latest technology and products.  Within Enterprise Analytics, the Platform & Solution Enablement (PSE) team defines the technical roadmap, builds the analytics foundation and enables early adoption of innovation out of SAPs product portfolio.  As a member of the PSE team, you will actively participate to the entire lifecycle of Business Intelligence (BI) and Enterprise Performance Management (EPM) solutions implementation. You will drive the setup, development and support of various components needed for best-in-class solutions while ensuring alignment with SAPs technical landscape and processes. Another key aspect of the position is closely interacting with other SAP IT teams, internal customers and also collaborating with the software development organization.  EXPECTATIONS AND TASKS:  Contribute to all BI and EPM technical project tasks, including requirements gathering, design, prototype, application development, testing and rollout Write and maintain clear, concise technical specifications for BI and EPM solutions Integrate work packages within new or existing processes, solutions and landscapes Proactively and frequently communicate with internal customers and stakeholders to build and maintain a solid working relationship Be committed to continuously improving your technical, leadership and teamwork skills. Be on the forefront of Analytics/BI and EPM trends Continuously seek opportunities for improving our processes, architecture and solutions Develop and foster partnership with Product development teams. Contribute / influence product features with ideas and proof of concepts based on business use cases EDUCATION AND QUALIFICATIONS / SKILLS AND COMPETENCIES  Bachelor's degree in Computer Science or Engineering 1-2 years working in a software development role Demonstrated experience with SQL scripting and database design/modeling techniques HANA experience is a plus Software coding/development hands on experience with JavaScript, HTML5 and CSS. Experience with SAP Products like Business Objects, BW, BPC or IP and ABAP is a plus Ability to work well in a team as well as independently and have a positive self-motivated can-do attitude Experience working in an Agile development environment, preferably SCRUM Strong analytical skills and ability to think in a complex and rapidly changing environment Excellent oral and written communication skills (English)",
        "Real Estate Marketing Associate A Real Estate Marketing Associate is a real estate agent who manages all aspects of a home purchase and sale. Buying or selling a home can be a daunting process for clients, so they depend on Real Estate Marketing Associates to give expert advice on how to market and sell their home and how to find a new home that is of good value and meets their needs. In this position, you will come up with marketing ideas to make your clients' home stand out amongst other homes so they can sell it quickly and for a favorable deal.  Job Responsibilities Lead clients through marketing their home to the local real estate community Stay informed on local home sales and new home listings Communicate with client portfolio to make sure all their real estate needs are being met Create marketing materials to advertise your real estate services Work with other Real Estate Marketing Associates to represent your clients during negotiations and the writing up of contracts Find appropriate homes to show your clients",
        "Marketing Assistant - Marketing Coordinator PURPOSE: Marketing and communication for company-Paid Training-Travel Opportunities-Management  MAJOR RESPONSIBILITY AREAS  · Implementation of marketing plans, including product positioning, campaign strategies, and market strategy insights.  · Discovery of strategic business opportunities through cross function collaboration with sales, HR, etc.  · Marketing opportunity for revenue  · Provide product/service support in order to establish proper channels of information and communication.  · Responsible for branding, advertising, trade shows, company events and promotional collateral  · Work with management on projects dealing with media relations, business communications, success stories  CORE COMPETENCIES:  These are personal traits that will best help the associate to successfully perform the essential functions of the job.  · Stress Tolerance - Job requires accepting criticism and dealing calmly and effectively with high stress situations.  · Judgment and Decision Making - Considering the relative costs and benefits of potential actions to choose the most appropriate one.  · Integrity - Job requires being honest and ethical.  · Initiative - Job requires a willingness to take on responsibilities and challenges.  · Leadership - Job requires a willingness to lead, take charge, and offer opinions and direction.  · Achievement/Effort - Job requires establishing and maintaining personally challenging achievement goals and exerting effort toward mastering tasks.  · Dependability - Job requires being reliable, responsible, and dependable, and fulfilling obligations.  · Social Orientation - Job requires preferring to work with others rather than alone, and being personally connected with others on the job.  · Attention to Detail - Job requires being careful about detail and thorough in completing work tasks.  · Cooperation - Job requires being pleasant with others on the job and displaying a good-natured, cooperative attitude.  · Candidate must be very articulate, have a sense of humor, easygoing, but very disciplined. We need a culture fit!  Requirements  ENTRY QUALIFICATIONS  · Bachelor's degree in Marketing, Communications, Advertising or Journalism  · Minimum (0) zero to (5) five years of relevant experience in marketing management with proven success, however we offer paid training  · Must have wide range of experience and understanding of the marketing including product positioning, pricing, promotions, market research, sales and distribution.  · Should be a proactive self-starter with the ability to work independently. Need strong ability to set priorities, solve problems, and be resourceful under pressure.  · Experience working with agency/client partners, exhibiting the ability to generate maximum return through effective marketing strategies and direction.",
        "Professor - Electrical Engineering & Computer Science The University of Cincinnati (UC), College of Engineering and Applied Science invites applications for multiple faculty positions in several departments in the College. Rank, tenure, salary and startup funding will be negotiated commensurate with the candidates qualifications and experience.  Applicants to Requisition # 11817will be considered for a position in the Department of Electrical Engineering and Computing Systems (EECS)  The Department of EECS offers ABET accredited undergraduate programs in Electrical Engineering, Computer Engineering and Computer Science. The department also has graduate programs that confer Master of Science, Master of Engineering, and Doctor of Philosophy degrees.  The successful candidates will be expected to teach graduate and undergraduate engineering courses specific to their specialty area online and/or in a traditional classroom setting, participate in service related activities, develop an externally funded research program in the applicable engineering field of expertise, advise graduate and undergraduate students, and publish research results in professional journals. In addition, the successful candidate may participate in curriculum and course development. Duties also include teaching courses in Electrical Engineering, Computer Engineering and /or Computer Science.  MINIMUM QUALIFICATIONS:PhD in Electrical Engineering, Computer Science, Computer Engineering or related field is required.",
        "Software Engineer Amdocs is seeking a Software Engineer to handle production issues, handle tickets, job failures and business related issues.  Successful candidate will also mitigate on time production issues, on-time to meet defined SLAs, and support and collaborate with interfacing applications.  Defines site objectives by analyzing user issues, envisioning system features and functionality. Troubleshooting development and production problems across multiple environments and operating platforms. Work on rapid development of urgent requirements by coordinating requirements, schedules, and activities; contributing to team meetings. Supports users by developing documentation and assistance tools. Support continuous improvement by investigating alternatives and technologies and presenting these for architectural review. Develop according to Amdocs standards to ensure that all milestone deliverables meet with defined standards. Deliverables include technical specifications, release notes, installation guides and other documentation. All you need is... Bachelor's degree in Science/IT/Computing or equivalent 3+ years of experience with ORACLE, SQL, Java/C and UNIX/Linux Sh Scripting Must have excellent English communication skills (Spanish speaking is advantage) Mandatory: 2+ years of experience developing and supporting large scale application Mandatory: 2+ Years of Web logic and Oracle hands on experience Mandatory: Ability to travel to the US for training and KT purposes Affluent in tools and strong working knowledge in automation techniques Understanding of interaction and interfaces between multiple billing systems and ability to interact with customer and business organizations Ability to support multiple projects, applications and collaborative environment                         Telecom billing and rating support experience Amdocs billing products knowledge is advantage Preferred: 1+ years of Agile/scrum experience Preferred: 1+ years of experience working with any ATT projects.",
        "Software Engineer Collaborate in deriving and implementing requirements for software (SW) elements, and in programming software under direction  Supports deriving and implementing requirements for SW subcomponents  o Support developing SW features and requirements analysis for product-related SW subcomponents including system interfaces on the basis of customer / market requirements (functional range, costs etc.)  o Support the implementation of requirements into requirement specifications.  o Develop data and functional models for comprehensive problems in the context of predetermined concepts.  o Define determining factors and data structures.  o Integrate functions specific to the product with existing internal and external systems (e.g. in the control base of different manufacturers  Support developing SW subcomponents  o Support developing SW design, if necessary also taking inter-connected functions and existing dependencies to other subcomponents into account, or rather the integration into an overall system while keeping the operational development guidelines in mind.  o Collaborate in preparing design and interface specification  Program software  Support creating testing specifications  Test software  o Test subcomponents within the overall software.  o Commission and release subcomponents  Maintain and further develop SW subcomponents  o Serial and customer support of delivered products. Support issues and adjustments.  o Optimize SW subcomponents. Take impact on functionality and efficiency of the overall system into account. Pursue technical development in the specialist areas and propose further developments.  Prepare documents  o Prepare all relevant documents according to process requirements (e.g. review of development steps and description of issue and change status).  Basic Qualifications:  Bachelors Degree in Computer Science, Electrical Engineering, Computer Engineering - 1-3 years of embedded software development or testing. - 1+ years of C, C++, Java, or other structured language. - Experience developing and debugging software in a real-time, embedded, multiprocessor, multi-interface environment.  Desired Qualifications:  -1-2+ years of low-level programming, experience to include boot loader, kernel, drivers and chipset.  Experience working with Android framework  Knows fundamental engineering concepts, practices and procedures  Demonstrated ability to develop and integrate software, prepare and present in design/code reviews and follow established processes for infotainment products."
        "Junior Trader In this role, you close business. Parchem’s Junior Traders build relationships with decision makers, develop large-scale clients, and use supply chain awareness to efficiently track deadlines. The most important characteristic of our sales specialist team is that we love helping our customers find what they are looking for. You are a fit if you are experienced, dynamic, self-motivated and disciplined with a strong ability to manage complex projects.    You are a fit if you have:  ·         A Bachelor’s Degree. Preferred: Business Administration, Economics, Engineering, Biology, Chemistry  ·         2+ years of experience in sales, marketing, purchasing or supply chain  ·         Experience conducting business over the phone with exemplary verbal and written communication skills  ·         Excellent ability to prioritize work while maintaining careful attention to detail  ·         Drive to succeed and to contribute to the growth of an exciting, fast-paced company  At Parchem, you will:  ·         Acquire, nurture, retain, and grow your business portfolio  ·         Gain an expertise in a variety of chemical industry sectors  ·         Conduct market research, stay on top of trends, and sell fine and specialty raw material within targeted industries  ·         Qualify hot leads  ·         Convert qualified leads into sales  ·         Anticipate customer demand and purchasing patterns to drive sales  ·         Build, negotiate, and heighten relationships with vendors  ·         Meet deadlines and respond  to business needs in a preemptive and professionally persistent manner"
        ]
jobD = cleanTitle(jobD)

#personal description
#jobRank = calcDescriptOverlapScore(pD, jobD)

#Test case candidate:----------------------------------------------------------------
candidateT = "Ruby On Rails Developer"
candidateComp = "Dell"
candidateD = "Ruby On Rails at Dell, Computer Software, Virtualization Data Center Ruby C Ruby on Rails C++ VMware Amazon Web Services MongoDB Hyper-V Java Linux"

#Computation---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

#next title model-------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

    #query
queryTitle = "SELECT A_experience_title, B_experience_title FROM [extern_job_match.rosanna_nextjob] "+"WHERE A_experience_title != 'null' AND B_experience_title != 'null' "+"LIMIT 50000"



#next company model-------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

queryCompany = "SELECT moveto, movefrom FROM [companies.employeemovementfromto] "+"WHERE moveto != 'null' AND movefrom != 'null' "+"LIMIT 50000"





# In[35]:

#next title model-------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

modelTitle = genPredictModel(queryTitle)     #model
uTitle = modelTitle[1]

#next company model-------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

modelCompany = genPredictModel(queryCompany) #model
uCompany = modelCompany[1]


# In[39]:

#Vectorize words to array withe memory-----------------------------------------------------

x = cleanTitle(convertData(queryTitle)[1])


def vectorize2(x, x1):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    
    vecX = vectorizer.fit_transform(x)
    #convert to array
    
    x1_new = vectorizer.transform(x1)
    
    vecX1 = x1_new.toarray()

    print('Finished vectorizer...', vecX[:1], vecX1)
    return vecX1

candidateT = vectorize2(x, candidateT)



# In[45]:

xC = cleanTitle(convertData(queryCompany)[1])
candidateComp = vectorize2(xC, candidateComp)


# In[46]:

probdictTitle = predictDictProb(modelTitle[0], candidateT, uTitle) #predicted prob
probdictCompany = predictDictProb(modelCompany[0], candidateComp, uCompany) #predicted prob dict


# In[91]:

#Start predicting------------------------------------------------------------

titleProb = probSimScore(jobT, probdictTitle) #score of next title and title list  similarity
companyProb = probSimScore(jobC, probdictCompany) #score of next company and company list  similarity


probScoreListCompany = probScore (companyProb, probdictCompany) #score of company
probScoreListTitle = probScore (titleProb, probdictTitle) #score of title


#description probability
descriptionList = calcDescriptOverlapScore(candidateD, jobD)

#Compiling them together
scoreArray = sortedJobNum (descriptionList, probScoreListTitle, probScoreListCompany)
total = top10Jobs (scoreArray, jobT)

#returns final rank list
print ('This is the final job rank list based on the test cases:', total)


# In[ ]:



