package session

import (
	"fmt"
	"log"
	"os"
)

// Internal data types to hold session, model, dataset, result, and task data during runtime.
type Session struct {
	Models   []Model
	Datasets []Dataset
	Results  []Result
	Tasks    []Task
}

type ModelDetails struct {
	Type              string
	AUCROC            float64
	TrainTime         float64
	TestTime          float64
	AveTreeDepth      float64
	AveLeafCount      float64
	AveSplitNodeCount float64
	AveTreeSize       float64
}

type Model struct {
	Name           string
	ID             string
	TrainedDataset string
	ID_num         int
	OptPath        string
	ClascPath      string
	Models         []ModelDetails
}

type Dataset struct {
	Name       string
	ID         string
	Datapoints int
	ID_num     int
	Path       string
}

type Result struct {
	ModelID   string
	DatasetID string
	Tree      string
	Precision string
	Recall    string
}

type Task struct {
	ID      string
	ModelID string
	Status  string
	ID_num  int
}

// Validation structures to capture our incoming TOML data
type TrainingConfig struct {
	Name                  string `form:"name" toml:"name" binding:"required"`
	DatasetID             string `form:"datasetid" toml:"dataset_id"`
	Dataset               string `form:"dataset" toml:"dataset"`
	DatasetHeader         string `form:"dataset_header" toml:"dataset_header"`
	TestAnomClass         string `form:"testanomclass" toml:"testanomclass"`
	TestNomlClass         string `form:"testnomlclass" toml:"testnomlclass"`
	Task                  string `form:"task" toml:"task"`
	Compare               bool   `form:"compare" toml:"compare"`
	GenerateOptmizedModel bool   `form:"generate_optimized_model" toml:"generate_optimized_model"`
	GenerateClassicModel  bool   `form:"generate_classic_model" toml:"generate_classic_model"`
	ModelPath             string `form:"path" toml:"path"`
	ModelName             string `form:"saved_model_name" toml:"saved_model_name" binding:"required"`
}

type UploadConfig struct {
	Name string `form:"name" toml:"name" binding:"required"`
}

type InferConfig struct {
	ModelID   string `form:"modelid" toml:"modelid" binding:"required"`
	DatasetID string `form:"datasetid" toml:"datasetid" binding:"required"`
	Dataset   string `form:"dataset" toml:"dataset"`
}

type DownloadConfig struct {
	ModelID   string `form:"modelid" toml:"modelid" binding:"required"`
	ModelType string `form:"modeltype" toml:"modeltype"`
}

// Structs for our structured responses to the client
type TrainingResponse struct {
	Response string
}

type InferenceResponse struct {
	TrainedPrecision float64
	TrainedRecall    float64
}

// Struct for holding toml info to submit to the training script
type ISOForestTrainingConfig struct {
	TaskType              string `toml:"task"`
	Dataset               string `toml:"dataset"`
	DatasetPath           string `toml:"dataset_path"`
	DatasetHeader         string `toml:"dataset_header"`
	TestAnomClass         string `form:"testanomclass" toml:"testanomclass"`
	TestNomlClass         string `form:"testnomlclass" toml:"testnomlclass"`
	Compare               bool   `form:"compare" toml:"compare"`
	GenerateOptmizedModel bool   `form:"generate_optimized_model" toml:"generate_optimized_model"`
	GenerateClassicModel  bool   `form:"generate_classic_model" toml:"generate_classic_model"`
	ModelPath             string `form:"path" toml:"path"`
	ModelName             string `toml:"saved_model_name"`
}

func (self *Session) Setup(volumePath string) {
	// Check along the volumePath for any existing models, datasets
	models_exists := false
	datasets_exists := false
	files, err := os.ReadDir(volumePath)
	if err != nil {
		log.Print(err)
	} else {
		for _, file := range files {
			if file.Name() == "models" {
				models_exists = true
				models, err := os.ReadDir("models")
				if err != nil {
					log.Print(err)
				}
				for i, model := range models {
					self.Models = append(self.Models, Model{model.Name(), "m" + fmt.Sprint(i), "Unknown", i, volumePath + "models" + model.Name() + ".model", volumePath + "models" + model.Name() + ".model", []ModelDetails{}})
				}
			}
			if file.Name() == "datasets" {
				datasets_exists = true
				datasets, err := os.ReadDir("datasets")
				if err != nil {
					log.Print(err)
				}
				for i, dataset := range datasets {
					self.Datasets = append(self.Datasets, Dataset{dataset.Name(), "d" + fmt.Sprint(i), 0, i, volumePath + "datasets" + dataset.Name() + ".csv"})
				}
			}
		}
	}

	if !models_exists {
		err := os.Mkdir(volumePath+"/models", 0777)
		if err != nil && !os.IsExist(err) {
			log.Print(err)
		}
	}

	if !datasets_exists {
		err := os.Mkdir(volumePath+"/datasets", 0777)
		if err != nil && !os.IsExist(err) {
			log.Print(err)
		}
	}
}
