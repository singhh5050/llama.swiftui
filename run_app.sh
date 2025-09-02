#!/bin/bash

# Script to run llama.swiftui app without translocation issues
# This helps avoid the macOS security warnings

echo "Building and running llama.swiftui app..."

# Change to the project directory
cd "$(dirname "$0")"

# Build the app
echo "Building the app..."
xcodebuild -project llama.swiftui.xcodeproj -scheme llama.swiftui -configuration Debug -derivedDataPath ./build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Find the built app
    APP_PATH=$(find ./build -name "llama.swiftui.app" -type d | head -1)
    
    if [ -n "$APP_PATH" ]; then
        echo "Found app at: $APP_PATH"
        
        # Remove quarantine attributes to avoid translocation
        echo "Removing quarantine attributes..."
        xattr -dr com.apple.quarantine "$APP_PATH" 2>/dev/null || true
        
        # Run the app
        echo "Launching app..."
        open "$APP_PATH"
    else
        echo "Error: Could not find built app"
        exit 1
    fi
else
    echo "Build failed!"
    exit 1
fi


