"""Loader stubs for real-world wildfire simulations and observational sensor APIs."""
import pandas as pd
from typing import Optional, List, Dict, Tuple, Any

class FarsiteLoaderStub:
    """Loader interface for importing FARSITE surface fire spread simulation outputs.
    
    FARSITE (Fire Area Simulator) simulates 2D fire behavior across complex landscapes.
    In future research phases, this loader will convert rasterized fire front progression
    (.fcf / .trj outputs) into per-node synthetic sensor signatures based on proximity
    to the flaming front.
    """
    @staticmethod
    def load_scenario(farsite_dir: str, node_coords: Dict[int, Tuple[float, float]]) -> Dict[int, pd.DataFrame]:
        """Convert FARSITE output rasters to schema-compatible IoT time series per node.
        
        Args:
            farsite_dir: Path to directory containing FARSITE raster exports (.fcf/.flm).
            node_coords: Mapping of node_id -> (latitude, longitude) or (x_m, y_m).
            
        Returns:
            Dictionary mapping node_id -> schema-compatible DataFrame with pre-ignition signatures.
        """
        raise NotImplementedError(
            "FARSITE live binary integration is an optional stretch goal. "
            "Use src.data.generator.WildfireDataGenerator for simulated experiments."
        )


class NASAFirmsLoaderStub:
    """Loader interface for NASA FIRMS (Fire Information for Resource Management System).
    
    In future operational iterations, this class connects to NASA's LANCE FIRMS REST API
    to pull VIIRS / MODIS thermal hotspot detections within specified spatial bounding boxes
    to act as ground-truth labels or exogenous early-warning signals for the GNN belief maps.
    """
    @staticmethod
    def fetch_thermal_anomalies(
        lat_min: float, lat_max: float, lon_min: float, lon_max: float, date_range_str: str, api_key: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch thermal anomalies within spatial bounding box.
        
        Returns:
            DataFrame with columns: ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'confidence'].
        """
        raise NotImplementedError(
            "Live NASA FIRMS API calls are out of scope for the simulated capstone build. "
            "Refer to offline synthetic fire labels in Parquet exports."
        )


class NOAAWeatherLoaderStub:
    """Loader interface for NOAA National Centers for Environmental Information (NCEI) weather data.
    
    Used to initialize real-world baseline macroclimate distributions (temperature, humidity, wind vector)
    for specific geographic wildfire zones.
    """
    @staticmethod
    def fetch_hourly_weather(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical hourly weather station readings."""
        raise NotImplementedError(
            "Live NOAA weather API integration is stubbed. "
            "Using deterministic AR(1) diurnal microclimate simulation in WildfireDataGenerator."
        )
