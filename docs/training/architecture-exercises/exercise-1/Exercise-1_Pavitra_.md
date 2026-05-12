# Exercise 1 – Buying Coffee with a Contactless Card at a Café
## Architecture Modelling using ArchiMate
| Author | P PAVITRA                                   |
|---|---------------------------------------------|
| Date | 11-May-2026                                 |
| Exercise | Enterprise Architecture Modelling with ArchiMate |
| System | Café Contactless Payment and Coffee Ordering System |
---
# System Overview
The Café Contactless Payment System is designed using the ArchiMate enterprise architecture framework. The system models the complete workflow involved in coffee ordering, contactless card payment processing, inventory handling, receipt generation, transaction storage, and cloud backup operations.
The architecture supports two major business actors:
- Customer – Places coffee orders and makes payment using a contactless card.
- Barista – Handles customer orders and payment processing.
The core business service of the system is the Coffee Ordering Service, which manages customer ordering activities and initiates payment processing through the Take Payment process.
The architecture integrates Business, Application, and Technology layers to represent the interaction between users, software systems, and infrastructure components.
---
# Business Layer
The Business Layer models the business actors, services, processes, and business objects involved in café operations.
## Business Actors
### Customer
Represents the customer who places coffee orders and pays using a contactless card.
### Barista
Represents the café employee responsible for handling orders and processing payments.
---
## Business Services and Processes
### Coffee Ordering Service
The primary business service responsible for managing coffee orders.
### Take Payment
Business process responsible for handling contactless card payment transactions after the order is placed.
---
## Business Objects
### Customer Order
Stores customer order information such as coffee items and quantities.
### Payment Receipt
Represents the receipt generated after successful payment completion.
---
## Business Layer Relationships
### Serving Relationship
- Coffee Ordering Service → Customer
The service is provided to the customer.
### Assignment Relationship
- Barista → Coffee Ordering Service
The barista performs and manages the service operation.
### Triggering Relationship
- Coffee Ordering Service → Take Payment
The payment process starts after order completion.
### Access Relationship
- Coffee Ordering Service → Customer Order
- Take Payment → Payment Receipt
The business processes access and generate business data objects.
---
# Application Layer
The Application Layer models the software applications and supporting services used in the café POS system.
## Application Components
### POS Application
The main application component responsible for:
- Managing orders
- Processing contactless payments
- Connecting inventory services
- Handling receipt generation
### Order Management System
Maintains and manages customer orders and transaction records.
---
## Application Services
### Receipt Service
Provides receipt generation functionality.
### Inventory Service
Maintains coffee stock and inventory availability information.
### Payment Gateway
Processes secure contactless card payment transactions.
---
## Data Object
### Payment Data
Stores payment-related transaction information processed by the system.
---
## Application Layer Relationships
### Serving Relationships
- Receipt Service → POS Application
- Inventory Service → POS Application
- Payment Gateway → POS Application
- Payment Gateway → Take Payment
These services support the POS application and payment process.
### Flow Relationships
- POS Application → Order Management System
- Payment Gateway → Order Management System
- Order Management System → Payment Data
Information and transaction data flow between application components.
### Realization Relationship
- POS Application → Coffee Ordering Service
The POS application implements the business service.
---
# Technology Layer
The Technology Layer models the infrastructure and storage components supporting the application environment.
## Technology Components
### POS Terminal
Physical terminal used by staff for billing and contactless card payment processing.
### Payment Server
Handles secure payment communication and transaction processing.
### Transaction Database
Stores payment records and customer transaction details.
### Cloud Backup Server
Maintains secure backup copies of transaction data.
---
## Technology Layer Relationships
### Flow Relationships
- POS Terminal → Transaction Database
- Payment Server → Transaction Database
- Transaction Database → Cloud Backup Server
Transaction and backup data flow through infrastructure components.
---
# Relationship Types Used
| Relationship | Purpose |
|---|---|
| Serving | One component provides service to another |
| Assignment | Actor performs a business process or service |
| Triggering | One process initiates another process |
| Access | Processes access or generate business data |
| Flow | Information or transaction data transfer |
| Realization | Application component implements business service |
---
# Architecture Flow Summary
1. The Customer places a coffee order using the Coffee Ordering Service.
2. The Barista manages ordering and payment activities.
3. The Coffee Ordering Service triggers the Take Payment process.
4. The POS Application coordinates ordering, payment, inventory, and receipt services.
5. The Payment Gateway securely processes contactless card payment transactions.
6. Order and payment information flow to the Order Management System.
7. Transaction details are stored in the Transaction Database.
8. Backup data is transferred to the Cloud Backup Server for reliability and recovery.
---
# Conclusion
This ArchiMate model demonstrates a layered enterprise architecture for a Café Contactless Payment and Coffee Ordering System. The architecture clearly represents business workflows, application interactions, infrastructure components, and cross-layer integrations using standard ArchiMate modelling concepts.
The system architecture is modular, scalable, and aligned with enterprise architecture modelling standards.