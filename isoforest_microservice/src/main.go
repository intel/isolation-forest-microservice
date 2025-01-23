package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/BurntSushi/toml"
	"github.com/gin-gonic/gin"
	session "intel.com/isoforest-microservice/session"
)

// Session object
var current_session session.Session

// setupRouter: Sets up the Gin-based http router with our options and our routes.
func setupRouter() *gin.Engine {
	router := gin.Default()

	//GET Methods
	router.GET("/status", getStatus)
	router.GET("/datasets", getDataset)
	router.GET("/models", getModel)
	//	router.GET("/models/tree", getModelTree)
	router.GET("/results", getResults)
	//POST Methods
	router.POST("/train", startTraining)
	router.POST("/data/upload", uploadData)
	//	router.POST("/model/upload", uploadModel)
	router.POST("/infer", infer)
	return router
}

// getStatus: Returns a list of: running + finished tasks; uploaded datasets; built models
func getStatus(c *gin.Context) {
	// Without ID, return everything. Begin building our return: start by querying the available task list and their status
	c.JSON(http.StatusOK, current_session)
	// Query the uploaded datasets

	// Query the available models

	// Return our list as a HTTP StatusOK

	// With ID, only return status of training job
}

// getDataset: Returns a list of available datasets in the microservice, or if an ID is provided returns information about a specific dataset
func getDataset(c *gin.Context) {
	// With no ID, returns a list of datasets available.
	c.JSON(http.StatusOK, current_session.Datasets)
	// If ID present, returns info about that dataset
}

// getModel: Returns a specific model based on the provided model ID. Without a model ID, it returns a list of available models
func getModel(c *gin.Context) {
	// If ID not present in request, query list of available models and return
	var downloadConfig session.DownloadConfig
	if err := c.BindTOML(&downloadConfig); err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
		return
	}
	var model session.Model
	for _, mod := range current_session.Models {
		if downloadConfig.ModelID == mod.ID {
			model = mod
			log.Printf("Downloading model %s...", model.ID)
			if downloadConfig.ModelType == "" || downloadConfig.ModelType == "QUANTIZED" {
				optmodelpath := "/storage/models/" + model.Name + "_" + "QUANTIZED.model"
				file, err := os.ReadFile(optmodelpath)
				if err != nil {
					c.AbortWithError(http.StatusBadGateway, err)
				}
				c.Data(200, "application/octet-stream", file)
				return
			}
			if downloadConfig.ModelType == "RANDOM" {
				file, err := os.ReadFile(model.ClascPath)
				if err != nil {
					c.AbortWithError(http.StatusBadGateway, err)
				}
				c.Data(200, "application/octet-stream", file)
				return
			}
		}
	}
	if model.ID == "" {
		log.Printf("Model not found: %s\nReturning models list", downloadConfig.ModelID)
	}
	c.JSON(http.StatusOK, current_session.Models)
	// If ID present, check if model exists. If yes, return model details (time of creation, dataset used, size)

	// If ID present and not exists, return error to user (http.StatusNotFound))

}

func getModelTree(c *gin.Context) {
	var downloadConfig session.DownloadConfig
	if err := c.BindTOML(&downloadConfig); err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
		return
	}
	var model session.Model
	for _, mod := range current_session.Models {
		if downloadConfig.ModelID == mod.ID {
			model = mod
			treesTOMLPath := generateTreesTOML(model.OptPath, "show_trees")
			channel_status := make(chan []byte)
			go func() {
				cmd := exec.Command("python3", "../..//iso_forest/main.py", treesTOMLPath)
				log.Println("Showing model tree...")
				out, err := cmd.CombinedOutput()
				if err != nil {
					log.Println(err)
				}
				current_session.Tasks[len(current_session.Tasks)-1].Status = "Complete"
				channel_status <- out
			}()
			response := strings.Split(strings.TrimSuffix(string(<-channel_status), "\n"), "\n")
			c.JSON(http.StatusOK, response)
			return

		}
	}
	if model.ID == "" {
		log.Printf("Model not found: %s\nReturning models list", downloadConfig.ModelID)
	}
	c.JSON(http.StatusOK, current_session.Models)
}

// getResults: Returns a specfiic inference job's results based on the provided run ID. Without a run ID, it returns a list of available result runs
func getResults(c *gin.Context) {
	// If job ID not present, return list of available results
	c.JSON(http.StatusOK, current_session.Results)
	// If ID is present, check if run exists, and then return the run results (model used, dataset trained with, dataset ran, results from run)

	// If ID present and not exists, return error ot user (http.StatusNotFound)
}

// startTraining: Based on an input TOML file, builds a new model and assigns it an ID.
func startTraining(c *gin.Context) {
	// If TOML provided, check for valid dataset ID. If everything's ready, start a training task as a trackable async goroutine. Add said goroutine to the list of tasks
	// While job runs in background, return list of features from dataset and number of datapoints being trained.
	log.Println("Building new model...")
	// Check TOML:
	var training_body session.TrainingConfig
	if err := c.BindTOML(&training_body); err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
		return
	}
	// Ready? Get model information ready, set new ID for this job, and add it to the list. Set status to "Getting Ready"
	// New Model
	var existing_model_ids []int
	max_id := 0
	for _, model := range current_session.Models {
		existing_model_ids = append(existing_model_ids, model.ID_num)
	}
	if len(existing_model_ids) != 0 {
		max_id = slices.Max(existing_model_ids)
	}
	var new_model session.Model
	new_model.ID_num = max_id + 1
	new_model.ID = "m" + fmt.Sprint(new_model.ID_num)
	new_model.Name = training_body.Name
	new_model.TrainedDataset = training_body.Dataset
	new_model.OptPath = "/storage/models/" + new_model.Name + "_QUANTIZED.model"
	new_model.ClascPath = "/storage/models/" + new_model.Name + "_RANDOM.model"
	// New Task
	var existing_task_ids []int
	for _, task := range current_session.Tasks {
		existing_task_ids = append(existing_task_ids, task.ID_num)
	}
	if len(existing_task_ids) != 0 {
		max_id = slices.Max(existing_task_ids)
	}
	var new_task session.Task
	new_task.ModelID = new_model.ID
	new_task.ID_num = max_id + 1
	new_task.ID = "t" + fmt.Sprint(new_task.ID_num)
	new_task.Status = "Running"
	current_session.Tasks = append(current_session.Tasks, new_task)
	log.Println("new task created")
	// Get our train config TOML ready - get the dataset path, get the features, get the data, get the name to set the path
	// Prep the environment and send the config to the training tool. Start a goroutine to handle running and set status to "running"
	dataset_path := ""
	if training_body.Dataset == "CUSTOM" {
		for _, dataset := range current_session.Datasets {
			if dataset.ID == new_model.TrainedDataset {
				dataset_path = dataset.Path
				break
			}
			response := "dataset not found, id: " + new_model.TrainedDataset
			c.JSON(http.StatusBadRequest, response)
		}
	} else {
		dataset_path = training_body.Dataset
	}
	// Read in our CSV file
	trainingtomlpath := generateTrainingTOML(dataset_path, dataset_path, training_body.TestAnomClass, training_body.TestNomlClass, new_model.OptPath, training_body.Task, training_body.Compare, training_body.GenerateOptmizedModel, training_body.GenerateClassicModel, training_body.ModelName)
	channel_status := make(chan []byte)
	go func() {
		cmd := exec.Command("python3", "../..//iso_forest/main.py", trainingtomlpath)
		log.Println("Starting training...")
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Println(err)
		}
		current_session.Tasks[len(current_session.Tasks)-1].Status = "Complete"
		channel_status <- out
	}()
	response := strings.Split(strings.TrimSuffix(string(<-channel_status), "\n"), "\n")
	log.Println(response)

	var optmodel session.ModelDetails
	// Take from our output from our script and fill in our model details
	optmodel.Type = strings.Split(response[len(response)-10], " ")[len(strings.Split(response[len(response)-10], " "))-1]
	optmodel.AUCROC, _ = strconv.ParseFloat(strings.Split(response[len(response)-8], " ")[len(strings.Split(response[len(response)-8], " "))-1], 64)
	optmodel.TrainTime, _ = strconv.ParseFloat(strings.Split(response[len(response)-7], " ")[len(strings.Split(response[len(response)-7], " "))-1], 64)
	optmodel.TestTime, _ = strconv.ParseFloat(strings.Split(response[len(response)-6], " ")[len(strings.Split(response[len(response)-6], " "))-1], 64)
	optmodel.AveTreeDepth, _ = strconv.ParseFloat(strings.Split(response[len(response)-5], " ")[len(strings.Split(response[len(response)-5], " "))-1], 64)
	optmodel.AveLeafCount, _ = strconv.ParseFloat(strings.Split(response[len(response)-4], " ")[len(strings.Split(response[len(response)-4], " "))-1], 64)
	optmodel.AveSplitNodeCount, _ = strconv.ParseFloat(strings.Split(response[len(response)-3], " ")[len(strings.Split(response[len(response)-3], " "))-1], 64)
	optmodel.AveTreeSize, _ = strconv.ParseFloat(strings.Split(response[len(response)-2], " ")[len(strings.Split(response[len(response)-2], " "))-1], 64)

	log.Println(training_body)
	if training_body.Compare {
		var clascmodel session.ModelDetails
		clascmodel.Type = strings.Split(response[len(response)-19], " ")[len(strings.Split(response[len(response)-19], " "))-1]
		clascmodel.AUCROC, _ = strconv.ParseFloat(strings.Split(response[len(response)-17], " ")[len(strings.Split(response[len(response)-17], " "))-1], 64)
		clascmodel.TrainTime, _ = strconv.ParseFloat(strings.Split(response[len(response)-16], " ")[len(strings.Split(response[len(response)-16], " "))-1], 64)
		clascmodel.TestTime, _ = strconv.ParseFloat(strings.Split(response[len(response)-15], " ")[len(strings.Split(response[len(response)-15], " "))-1], 64)
		clascmodel.AveTreeDepth, _ = strconv.ParseFloat(strings.Split(response[len(response)-14], " ")[len(strings.Split(response[len(response)-14], " "))-1], 64)
		clascmodel.AveLeafCount, _ = strconv.ParseFloat(strings.Split(response[len(response)-13], " ")[len(strings.Split(response[len(response)-13], " "))-1], 64)
		clascmodel.AveSplitNodeCount, _ = strconv.ParseFloat(strings.Split(response[len(response)-12], " ")[len(strings.Split(response[len(response)-12], " "))-1], 64)
		clascmodel.AveTreeSize, _ = strconv.ParseFloat(strings.Split(response[len(response)-11], " ")[len(strings.Split(response[len(response)-11], " "))-1], 64)
		new_model.Models = append(new_model.Models, clascmodel)
	}
	new_model.Models = append(new_model.Models, optmodel)
	current_session.Models = append(current_session.Models, new_model)

	// Return a good status to the user.
	c.JSON(http.StatusOK, response)
}

// uploadData: Uploads a provided dataset (.csv) to the microservice datastore and assigns it an ID.
func uploadData(c *gin.Context) {
	// Check if dataset is a .csv. If valid (trust the user to add a valid dataset), extract the feature list and then assign it an ID for future use.

	file, err := c.FormFile("file")
	if err != nil {
		c.String(http.StatusBadRequest, "Form error %s", err.Error())
	}
	filename := filepath.Base(file.Filename)
	fmt.Println(filename)
	path := "/storage/datasets/"
	if err := c.SaveUploadedFile(file, "/storage/datasets/"+filename); err != nil {
		if err := c.SaveUploadedFile(file, "./"+filename); err != nil {
			c.String(http.StatusBadRequest, "Error uploading file: %s", err.Error())
			return
		}
		c.String(http.StatusOK, "storage mount not available, saving locally")
		path = "./"
	}
	var existing_dataset_ids []int
	max_id := 0
	for _, model := range current_session.Models {
		existing_dataset_ids = append(existing_dataset_ids, model.ID_num)
	}
	if len(existing_dataset_ids) != 0 {
		max_id = slices.Max(existing_dataset_ids)
	}
	fmt.Println("assinging dataset id: " + fmt.Sprint(max_id+1))
	// Give our session the new dataset
	var new_dataset session.Dataset
	new_dataset.ID_num = max_id + 1
	new_dataset.ID = "d" + fmt.Sprint(new_dataset.ID_num)
	new_dataset.Name = strings.TrimSuffix(filename, ".csv")
	new_dataset.Path = path + filename
	current_session.Datasets = append(current_session.Datasets, new_dataset)
	//Return good status
	c.JSON(http.StatusOK, new_dataset)
}

// uploadModel: Uploads a previously downloaded model to the microservice datastore, verifies it, and assigns it an ID
func uploadModel(c *gin.Context) {
	log.Println("Uploading Model...")
	file, err := c.FormFile("file")
	if err != nil {
		c.String(http.StatusBadRequest, "Form error %s", err.Error())
	}
	filename := filepath.Base(file.Filename)
	fmt.Println(filename)
	if err := c.SaveUploadedFile(file, "/storage/models/"+filename); err != nil {
		if err := c.SaveUploadedFile(file, "./"+filename); err != nil {
			c.String(http.StatusBadRequest, "Error uploading file: %s", err.Error())
			return
		}
		c.String(http.StatusOK, "storage mount not available, saving locally")
	}
	// New Model
	var existing_model_ids []int
	max_id := 0
	for _, model := range current_session.Models {
		existing_model_ids = append(existing_model_ids, model.ID_num)
	}
	if len(existing_model_ids) != 0 {
		max_id = slices.Max(existing_model_ids)
	}
	var new_model session.Model
	new_model.ID_num = max_id + 1
	new_model.ID = "m" + fmt.Sprint(new_model.ID_num)
	new_model.Name = filename
	new_model.TrainedDataset = "unknown"
	new_model.OptPath = "/storage/models/" + new_model.Name + "_QUANTIZED.model"
	new_model.ClascPath = "/storage/models/" + new_model.Name + "_RANDOM.model"
	current_session.Models = append(current_session.Models, new_model)
	// Return good status coode
	c.JSON(http.StatusOK, new_model)
}

// infer: infers on a defined dataset with a defined model, and returns the results.
func infer(c *gin.Context) {
	// take TOML with info on model and dataset. With only model, use the same dataset. Return results from inference
	// Load up our chosen model (make sure it exists, get its full path on disk),
	log.Println("Inferring with model...")
	var infer_body session.InferConfig
	if err := c.BindTOML(&infer_body); err != nil {
		c.AbortWithError(http.StatusBadRequest, err)
		return
	}
	//load up our dataset (if available - otherwise use the dataset used to train it. If that's not available in the model information, return an error)
	dataset := infer_body.Dataset
	dataset_path := ""
	for _, dataset := range current_session.Datasets {
		if dataset.ID == infer_body.DatasetID {
			dataset_path = dataset.Path
			break
		}
		response := "dataset not found, id: " + infer_body.DatasetID
		c.JSON(http.StatusBadRequest, response)
	}

	model_path := ""
	model_infer_name := ""
	for _, model := range current_session.Models {
		if model.ID == infer_body.ModelID {
			model_path = model.OptPath
			break
		}
		response := "dataset not found, id: " + infer_body.ModelID
		c.JSON(http.StatusBadRequest, response)
	}
	infertomlpath := generateInferenceTOML(dataset, dataset_path, model_path, model_infer_name, "infer")
	log.Println(infertomlpath)
	// Infer, store the results
	channel_status := make(chan []byte)
	go func() {
		cmd := exec.Command("python3", "/app/iso_forest/main.py", infertomlpath)
		log.Println("Starting inference...")
		output, err := cmd.Output()
		if err != nil {
			log.Println(err)
		}
		if len(current_session.Tasks) != 0 {
			current_session.Tasks[len(current_session.Tasks)-1].Status = "Complete"
		}
		channel_status <- output
	}()
	response := strings.Split(strings.TrimSuffix(string(<-channel_status), "\n"), "\n")
	log.Println(response)
	//trainedprecision, _ := strconv.ParseFloat(response[0], 64)
	//trainedrecall, _ := strconv.ParseFloat(response[1], 64)
	// Return a good status to the user.
	c.JSON(http.StatusOK, response) //session.InferenceResponse{TrainedPrecision: trainedprecision, TrainedRecall: trainedrecall})
	// Return the inference results and a good status code
}

func generateTrainingTOML(filepath string, dataset string, datasetanomclass string, datasetnomlclass string, modelpath string, tasktype string, compare bool, genopt bool, genclasc bool, modname string) string {
	path := "/storage/train.toml"
	trainingToml := session.ISOForestTrainingConfig{TaskType: tasktype, Dataset: dataset, DatasetPath: filepath, TestAnomClass: datasetanomclass, TestNomlClass: datasetnomlclass, ModelPath: modelpath, GenerateOptmizedModel: genopt, GenerateClassicModel: genclasc, Compare: compare, ModelName: modname}
	buf := new(bytes.Buffer)
	err := toml.NewEncoder(buf).Encode(trainingToml)
	if err != nil {
		log.Fatal(err)
	}
	// Write to file
	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		path = "./train.toml"
		f, err = os.Create(path)
		log.Println("storage volume not available, writing to local directory")
		if err != nil {
			log.Fatal(err)
		}

	}
	_, err = f.Write(buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	return path
}

func generateInferenceTOML(dataset string, datasetpath string, modelpath string, infername string, tasktype string) string {
	path := "/storage/infer.toml"
	trainingToml := session.ISOForestTrainingConfig{TaskType: tasktype, Dataset: dataset, DatasetPath: datasetpath, ModelPath: modelpath}
	buf := new(bytes.Buffer)
	err := toml.NewEncoder(buf).Encode(trainingToml)
	if err != nil {
		log.Fatal(err)
	}
	// Write to file
	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		path = "./infer.toml"
		f, err = os.Create(path)
		log.Println("storage volume not available, writing to local directory")
		if err != nil {
			log.Fatal(err)
		}

	}
	_, err = f.Write(buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	return path
}

func generateTreesTOML(modelpath string, tasktype string) string {
	path := "/storage/trees.toml"
	treesTOML := session.ISOForestTrainingConfig{TaskType: tasktype, ModelPath: modelpath}
	buf := new(bytes.Buffer)
	err := toml.NewEncoder(buf).Encode(treesTOML)
	if err != nil {
		log.Fatal(err)
	}
	// Write to file
	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		path = "./infer.toml"
		f, err = os.Create(path)
		log.Println("storage volume not available, writing to local directory")
		if err != nil {
			log.Fatal(err)
		}

	}
	_, err = f.Write(buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	return path
}

// main: our main function
func main() {
	gin.SetMode(gin.ReleaseMode)
	// Startup Tasks: Check for existing models, datasets in the mounted volume, add then to our model and dataset list
	volumePath := os.Getenv("VOLUMEPATH")
	if volumePath == "" {
		volumePath = "/storage"
		os.Mkdir("/storage/models", 0755)
		os.Mkdir("/storage/results", 0755)
	}
	//Create Router
	router := setupRouter()
	// Set up session variables
	current_session.Setup(volumePath)
	//Router Run
	router.Run(":9001")
}
