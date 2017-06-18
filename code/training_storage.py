"""
Contains classes and functions for creating, maintaining, and interacting
with a SQLite database containing data from training runs of various models.
"""

import os
from platform import processor
from datetime import datetime

from sqlalchemy import (create_engine, Column, Table, UniqueConstraint, 
                        ForeignKey, Integer, DateTime, Text, Boolean, Float)
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from psutil import virtual_memory
from GPUtil import getGPUs


MODEL_DIR = os.path.join(os.pardir, "models")
DATABASE_PATH = os.path.join(MODEL_DIR, "desc2code_training_database.db")
DATABASE_URL = "sqlite:///" + os.path.abspath(DATABASE_PATH).replace("\\\\", "/")
Base = declarative_base()


def _create_schema():
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)


def start_session():
    engine = create_engine(DATABASE_URL, echo=False)
    return sessionmaker(bind=engine)()


class Model(Base):
    __tablename__ = "Model"

    model_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    model_graph_file = Column(Text)

    # Model architecture
    description = Column(Text)
    input_type = Column(Text)
    output_type = Column(Text)
    encoder_cell = Column(Text)
    decoder_cell = Column(Text)
    layers = Column(Integer)
    bidirectional_encoder = Column(Boolean)
    attention = Column(Boolean)

    # Other hyperparameters
    hidden_dimension = Column(Integer)
    encoder_vocab_size = Column(Integer)
    decoder_vocab_size = Column(Integer)
    encoder_embedding_size = Column(Integer)
    decoder_embedding_size = Column(Integer)
    batch_size = Column(Integer)
    truncated_backprop = Column(Boolean)
    learning_rate = Column(Float)
    optimization_algorithm = Column(Text)

    __table_args__ = (UniqueConstraint(
        "description", "input_type", "output_type", "encoder_cell", 
        "decoder_cell", "layers", "bidirectional_encoder", "attention",
        "hidden_dimension", "encoder_vocab_size", "decoder_vocab_size",
        "encoder_embedding_size", "decoder_embedding_size", "batch_size",
        "truncated_backprop", "learning_rate", "optimization_algorithm"),)

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "model_graph_file": self.model_graph_file,
            "description": self.description,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "encoder_cell": self.encoder_cell,
            "decoder_cell": self.decoder_cell,
            "layers": self.layers,
            "bidirectional_encoder": self.bidirectional_encoder,
            "attention": self.attention,
            "hidden_dimension": self.hidden_dimension,
            "encoder_vocab_size": self.encoder_vocab_size,
            "decoder_vocab_size": self.decoder_vocab_size,
            "encoder_embedding_size": self.encoder_embedding_size,
            "decoder_embedding_size": self.decoder_embedding_size,
            "batch_size": self.batch_size,
            "truncated_backprop": self.truncated_backprop,
            "learning_rate": self.learning_rate,
            "optimization_algorithm": self.optimization_algorithm,
        }


class TrainingRun(Base):
    __tablename__ = "Training_Run"

    run_id = Column(Integer, primary_key=True)
    model_parameters_file = Column(Text)
    begin_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    used_gpus = Column(Integer)

    # Computer info
    cpu = Column(Text)
    gpu = Column(Text)
    ram = Column(Integer)

    model_id = Column(Integer, ForeignKey("Model.model_id"))
    model = relationship("Model", backref="training_runs")

    dataset_id = Column(Integer, ForeignKey("Dataset_Info.dataset_id"))
    dataset = relationship("DatasetInfo", backref="training_runs")


class DatasetInfo(Base):
    __tablename__ = "Dataset_Info"

    dataset_id = Column(Integer, primary_key=True)
    training_fraction = Column(Float)
    validation_fraction = Column(Float)

    # Dataset info
    input_sequences = Column(Integer)
    average_input_length = Column(Float)
    smallest_input_length = Column(Integer)
    largest_input_length = Column(Integer)

    output_sequences = Column(Integer)
    average_output_length = Column(Float)
    smallest_output_length = Column(Integer)
    largest_output_length = Column(Integer)


class ModelEvaluation(Base):
    __tablename__ = "Model_Evaluation"

    evaluation_id = Column(Integer, primary_key=True)
    dataset_type = Column(Text) # Training, validation, or test
    timestamp = Column(DateTime)
    loss = Column(Float)
    batch = Column(Integer)
    epoch = Column(Integer)
    step = Column(Integer)

    run_id = Column(Integer, ForeignKey("Training_Run.run_id"))
    training_run = relationship("TrainingRun", backref="evaluation_steps")


class InputExample(Base):
    __tablename__ = "Input_Example"

    input_id = Column(Integer, primary_key=True)
    vector = Column(Text)  # e.g. "12, 15, 1, 74"
    text = Column(Text)

    model_id = Column(Integer, ForeignKey("Model.model_id"))
    model = relationship("Model", backref="input_examples")


class TargetExample(Base):
    __tablename__ = "Target_Example"

    target_id = Column(Integer, primary_key=True)
    vector = Column(Text)  # e.g. "12, 15, 1, 74"
    text = Column(Text)

    input_id = Column(Integer, ForeignKey("Input_Example.input_id"))
    input = relationship("InputExample", backref="targets")

    model_id = Column(Integer, ForeignKey("Model.model_id"))
    model = relationship("Model", backref="target_examples")


class ModelOutputExample(Base):
    __tablename__ = "Model_Output_Example"

    output_id = Column(Integer, primary_key=True)
    vector = Column(Text)  # e.g. "12, 15, 1, 74"
    text = Column(Text)

    input_id = Column(Integer, ForeignKey("Input_Example.input_id"))
    input = relationship("InputExample", backref="model_outputs")

    model_id = Column(Integer, ForeignKey("Model.model_id"))
    model = relationship("Model", backref="output_examples")


def store_model_info(description, input_type, output_type, encoder_cell,
                     decoder_cell, layers, bidirectional_encoder, attention,
                     hidden_dimension, encoder_vocab_size, decoder_vocab_size,
                     encoder_embedding_size, decoder_embedding_size, batch_size, 
                     truncated_backprop, learning_rate, optimization_algorithm,
                     timestamp=datetime.utcnow(), graph_file_path=None):
    session = start_session()
    model = Model(description=description, 
                  input_type=input_type, 
                  output_type=output_type, 
                  encoder_cell=encoder_cell,
                  decoder_cell=decoder_cell, 
                  layers=layers, 
                  bidirectional_encoder=bidirectional_encoder, 
                  attention=attention, 
                  hidden_dimension=hidden_dimension, 
                  encoder_vocab_size=encoder_vocab_size, 
                  decoder_vocab_size=decoder_vocab_size,
                  encoder_embedding_size=encoder_embedding_size, 
                  decoder_embedding_size=decoder_embedding_size, 
                  batch_size=batch_size, 
                  truncated_backprop=truncated_backprop, 
                  learning_rate=learning_rate, 
                  optimization_algorithm=optimization_algorithm, 
                  timestamp=timestamp)
    if graph_file_path:
        model.model_graph_file = graph_file_path
    session.add(model)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        model_id = session.query(Model.model_id).filter(
            Model.description == description, Model.input_type == input_type,
            Model.output_type == output_type, Model.encoder_cell == encoder_cell,
            Model.decoder_cell == decoder_cell, Model.layers == layers,
            Model.bidirectional_encoder == bidirectional_encoder, 
            Model.attention == attention, Model.hidden_dimension == hidden_dimension, 
            Model.encoder_vocab_size == encoder_vocab_size, 
            Model.decoder_vocab_size == decoder_vocab_size,
            Model.encoder_embedding_size == encoder_embedding_size,
            Model.decoder_embedding_size == decoder_embedding_size,
            Model.batch_size == batch_size, 
            Model.truncated_backprop == truncated_backprop, 
            Model.learning_rate == learning_rate,
            Model.optimization_algorithm == optimization_algorithm).scalar()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store model!")
    else:
        model_id = model.model_id

    return model_id


def store_model_graph_file(model_id, graph_file_path):
    session = start_session()
    session.query(Model).filter(Model.model_id == model_id).update(
        {Model.model_graph_file: graph_file_path})
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store model TF graph file name!")


def store_training_run(model_id, parameters_file_path, begin_timestamp, 
                       input_sequences, average_input_length, 
                       smallest_input_length, largest_input_length, 
                       output_sequences, average_output_length, 
                       smallest_output_length, largest_output_length, 
                       end_timestamp=datetime.utcnow(), used_gpus=1, 
                       cpu=None, gpu=None, ram=None, training_fraction=0.8, 
                       validation_fraction=0.1):
    if cpu is None:
        cpu = processor()
    if ram is None:
        ram = int(round(virtual_memory().total / 1024**3))
    if gpu is None:
        gpu = getGPUs()[0].name

    session = start_session()
    
    dataset = DatasetInfo(training_fraction=training_fraction,
                          validation_fraction=validation_fraction,
                          input_sequences=input_sequences,
                          average_input_length=average_input_length,
                          smallest_input_length=smallest_input_length,
                          largest_input_length=largest_input_length,
                          output_sequences=output_sequences,
                          average_output_length=average_output_length,
                          smallest_output_length=smallest_output_length,
                          largest_output_length=largest_output_length)

    training_run = TrainingRun(begin_timestamp=begin_timestamp,
                               end_timestamp=end_timestamp,
                               used_gpus=used_gpus,
                               cpu=cpu, gpu=gpu, ram=ram)
    training_run.model_id = model_id
    training_run.dataset = dataset

    session.add(training_run)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store training run!")

    return training_run.run_id


def store_evaluation_track(run_id, dataset_type, evaluation_track):
    """
    Args:
        dataset_type: String, accepts 'training', 'validation', or 'test'.
        evaluation_track: List, contains elements of the form 
            (loss, timestamp, epoch, batch).
    """
    session = start_session()
    for step, (loss, timestamp, epoch, batch) in enumerate(evaluation_track):
        model_evaluation = ModelEvaluation(
            dataset_type=dataset_type, loss=loss, timestamp=timestamp, 
            epoch=epoch, batch=batch, step=step)
        model_evaluation.run_id = run_id
        session.add(model_evaluation)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store evaluation track!")


def store_model_run(model_output_vector, model_output_text, input_id=None, 
                    input_vector=None, input_text=None, target_vector=None, 
                    target_text=None, model=None, model_id=None):
    session = start_session()
    if model is not None:
        model_id = model.model_id

    output_example = ModelOutputExample(vector=model_output_vector,
                                        text=model_output_text)
    if input_id is not None:
        output_example.input_id = input_id
    elif input_vector is not None and input_text is not None:
        input_example = InputExample(vector=input_vector, text=input_text)
        input_example.model_id = model_id
        session.add(input_example)
        session.flush()
        input_id = input_example.input_id
        output_example.input_id = input_id

    if target_vector is not None and target_text is not None:
        target_example = TargetExample(vector=target_vector, text=target_text,
                                       model_id=model_id, input_id=input_id)
        session.add(target_example)

    session.add(output_example)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store model run!")


def get_model_info(most_recent=None, input_type=None, output_type=None, 
                   encoder_cell=None, decoder_cell=None, layers=None, 
                   bidirectional_encoder=None, attention=None, 
                   hidden_dimension=None, encoder_vocab_size=None,
                   decoder_vocab_size=None, encoder_embedding_size=None, 
                   decoder_embedding_size=None, batch_size=None, 
                   truncated_backprop=None, learning_rate=None, 
                   optimization_algorithm=None, timestamp=None, 
                   attained_training_loss=None, attained_validation_loss=None, 
                   attained_test_loss=None):
    session = start_session()
    query = session.query(Model)

    if most_recent is not None:
        return [query.order_by(Model.timestamp.desc()).all()[0],]
    if input_type is not None:
        query.filter(Model.input_type == input_type)
    if output_type is not None:
        query.filter(Model.output_type == output_type)
    if encoder_cell is not None:
        query.filter(Model.encoder_cell == encoder_cell)
    if decoder_cell is not None:
        query.filter(Model.decoder_cell == decoder_cell)
    if layers is not None:
        query.filter(Model.layers == layers)
    if bidirectional_encoder is not None:
        query.filter(Model.bidirectional_encoder == bidirectional_encoder)
    if attention is not None:
        query.filter(Model.attention == attention)
    if hidden_dimension is not None:
        query.filter(Model.hidden_dimension == hidden_dimension)
    if encoder_vocab_size is not None:
        query.filter(Model.encoder_vocab_size == encoder_vocab_size)
    if decoder_vocab_size is not None:
        query.filter(Model.decoder_vocab_size == decoder_vocab_size)
    if encoder_embedding_size is not None:
        query.filter(Model.encoder_embedding_size == encoder_embedding_size)
    if batch_size is not None:
        query.filter(Model.batch_size == batch_size)
    if truncated_backprop is not None:
        query.filter(Model.truncated_backprop == truncated_backprop)
    if learning_rate is not None:
        query.filter(Model.learning_rate == learning_rate)
    if optimization_algorithm is not None:
        query.filter(Model.optimization_algorithm == optimization_algorithm)
    if timestamp is not None:
        query.filter(Model.timestamp >= timestamp)
    if attained_training_loss is not None:
        query.join(TrainingRun).join(ModelEvaluation).filter(
            ModelEvaluation.dataset_type == "training", 
            ModelEvaluation.loss <= attained_training_loss)
    if attained_validation_loss is not None:
        query.join(TrainingRun).join(ModelEvaluation).filter(
            ModelEvaluation.dataset_type == "validation", 
            ModelEvaluation.loss <= attained_validation_loss)
    if attained_test_loss is not None:
        query.join(TrainingRun).join(ModelEvaluation).filter(
            ModelEvaluation.dataset_type == "test", 
            ModelEvaluation.loss <= attained_test_loss)

    return query.all()


def get_training_run_info(model_id=None, model=None):
    session = start_session()
    if model is not None:
        model_id = model.model_id
    try:
        training_run = session.query(TrainingRun).filter(
            TrainingRun.model_id == model_id).one()
    except NoResultFound:
        training_run = None

    return training_run


def get_evaluation_track(dataset_type, training_run_id=None, training_run=None):
    session = start_session()
    if training_run is not None:
        training_run_id = training_run.run_id
    evaluation_track = session.query(ModelEvaluation).join(
        TrainingRun).filter(TrainingRun.run_id == training_run_id, 
        ModelEvaluation.dataset_type == dataset_type).all()

    return evaluation_track

